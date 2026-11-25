//! Runs translated shaders on the CPU, so what a lowered instruction
//! computes can be checked on any host, without a GPU or `dxil.dll`.
//!
//! The DXIL inside a container is LLVM bitcode. LLVM's own tools read it:
//! `opt` gives it the host's data layout and triple (DXIL's pointers are 32
//! bits), `llvm-link` adds `cpu_runtime.ll`, which stands in for the
//! `dx.op` intrinsics, and `lli` runs it. The programs have the shape of
//! `valu_tests.rs`' (`build_valu_cs`): four root constants arrive in
//! `s0..s3`, are copied to `v1..v4`, the body leaves its result in `v0`, and
//! `v0` is stored to dword 0 of a raw UAV. So an expectation written here
//! holds on the GPU too, and the other way round.
//!
//! The tools are found on `PATH`, or in `VICTORY_LLVM_BIN` if set. Without
//! them, or on a host other than x86-64 Linux, every test here says it was
//! skipped (see [`super::test_skip`]).
//!
//! What this checks: the arithmetic of the translation. What it cannot: the
//! host compiler's and driver's own handling (denormal flushing, a
//! vendor's f16 rounding, wave operations across lanes, which are one lane
//! here). Those stay with `valu_tests.rs` on Windows.

use std::path::{Path, PathBuf};
use std::process::Command;

use super::lower;
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;
const RUNTIME: &str = include_str!("cpu_runtime.ll");
const LAYOUT: &str =
    "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128";
const TRIPLE: &str = "x86_64-unknown-linux-gnu";

/// The LLVM tools, or why they are not usable here.
struct Tools {
    opt: PathBuf,
    link: PathBuf,
    lli: PathBuf,
}

fn tools() -> Result<&'static Tools, &'static str> {
    static TOOLS: std::sync::OnceLock<Result<Tools, &'static str>> = std::sync::OnceLock::new();
    TOOLS
        .get_or_init(|| {
            if !cfg!(all(target_os = "linux", target_arch = "x86_64")) {
                return Err("CPU runs of translated shaders are set up for x86-64 Linux only");
            }
            // VICTORY_LLVM_BIN, else PATH, else the newest of Debian's and
            // Ubuntu's versioned installs (/usr/lib/llvm-N/bin).
            let mut dirs: Vec<Option<PathBuf>> = match std::env::var_os("VICTORY_LLVM_BIN") {
                Some(d) => vec![Some(PathBuf::from(d))],
                None => vec![None],
            };
            if dirs == [None] {
                let mut versioned: Vec<(u32, PathBuf)> = std::fs::read_dir("/usr/lib")
                    .into_iter()
                    .flatten()
                    .flatten()
                    .filter_map(|e| {
                        let n = e
                            .file_name()
                            .to_str()?
                            .strip_prefix("llvm-")?
                            .parse()
                            .ok()?;
                        Some((n, e.path().join("bin")))
                    })
                    .collect();
                versioned.sort();
                dirs.extend(versioned.into_iter().rev().map(|(_, d)| Some(d)));
            }
            let works = |path: &Path| {
                Command::new(path)
                    .arg("--version")
                    .output()
                    .is_ok_and(|o| o.status.success())
            };
            dirs.iter()
                .find_map(|dir| {
                    let at = |name: &str| {
                        dir.as_ref()
                            .map_or_else(|| PathBuf::from(name), |d| d.join(name))
                    };
                    let t = Tools {
                        opt: at("opt"),
                        link: at("llvm-link"),
                        lli: at("lli"),
                    };
                    (works(&t.opt) && works(&t.link) && works(&t.lli)).then_some(t)
                })
                .ok_or("LLVM's opt, llvm-link and lli were not found (set VICTORY_LLVM_BIN)")
        })
        .as_ref()
        .map_err(|e| *e)
}

/// SOP1 `s_mov_b32 sdst, src0`.
const fn s_mov_b32(sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | 0x03 << 8 | src0 as u32
}

/// VOP1 `op vdst, src0` (`src0`: 0..105 = SGPR, 256..511 = VGPR).
const fn vop1(op: u16, vdst: u8, src0: u16) -> u32 {
    0x7E00_0000 | (vdst as u32) << 17 | (op as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// VOP2 `op vdst, src0, vsrc1`.
const fn vop2(op: u8, vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    (op as u32 & 0x3F) << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// VOP3 `op vdst, src0, src1, src2` with `clamp` and `neg` (one bit per source).
const fn vop3(op: u16, vdst: u8, src: [u16; 3], neg: u8, clamp: bool) -> [u32; 2] {
    [
        0x35 << 26 | (op as u32 & 0x3FF) << 16 | (clamp as u32) << 15 | vdst as u32,
        (neg as u32 & 7) << 29
            | (src[0] as u32 & 0x1FF)
            | (src[1] as u32 & 0x1FF) << 9
            | (src[2] as u32 & 0x1FF) << 18,
    ]
}

/// VOP3P `op vdst, src0, src1, src2` (op_sel_hi on for every source).
const fn vop3p(op: u16, vdst: u8, src: [u16; 3], clamp: bool) -> [u32; 2] {
    [
        0x33 << 26 | (op as u32 & 0x7F) << 16 | (clamp as u32) << 15 | 1 << 14 | vdst as u32,
        (src[0] as u32 & 0x1FF)
            | (src[1] as u32 & 0x1FF) << 9
            | (src[2] as u32 & 0x1FF) << 18
            | 3 << 27,
    ]
}

const fn vgpr(n: u8) -> u16 {
    256 + n as u16
}

/// `v1`, `v2`, `v3`: the first three inputs.
const V123: [u16; 3] = [vgpr(1), vgpr(2), vgpr(3)];

/// The program around `body`, as `valu_tests.rs` builds it.
fn program(body: &[u32]) -> Vec<u32> {
    let mut words = vec![s_mov_b32(5, 0x80)];
    // s_load_dwordx4 s[8:11], s[4:5], 0: the UAV's descriptor.
    words.extend([0x3D << 26 | 0x02 << 18 | 8 << 6 | 2, 125 << 25]);
    for i in 0..4u8 {
        words.push(vop1(0x001, i + 1, u16::from(i))); // v_mov_b32 v(i+1), s(i)
    }
    words.extend_from_slice(body);
    // buffer_store_dword v0, off, s[8:11], 0
    words.extend([0x38 << 26 | 0x1C << 18, 0x80 << 24 | 2 << 16]);
    words.push(S_ENDPGM);
    words
}

/// The DXIL bitcode of `body` translated in [`program`].
fn translate(body: &[u32]) -> Result<Vec<u8>, String> {
    let code: Vec<u8> = program(body).iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1,
            threads: [1, 1, 1],
        }),
        code,
    };
    let e = |e: crate::ShaderError| e.to_string();
    let p = ir::build(&native.code, false, native.vgprs).map_err(e)?;
    let st = structure::structure(&p).map_err(e)?;
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).map_err(e)?;
    let io = io::raise(&p, &native).map_err(e)?;
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    })
    .map_err(e)?;
    bitcode(&lowered.dxil)
        .map(<[u8]>::to_vec)
        .ok_or_else(|| "no DXIL part".into())
}

/// The bitcode in a container's DXIL part: after the 8-byte program header,
/// a bitcode header whose third dword is the bitcode's offset from it and
/// whose fourth is its size.
fn bitcode(container: &[u8]) -> Option<&[u8]> {
    let c = crate::dxbc::Container::parse(container).ok()?;
    let d = c.part(crate::dxbc::FourCC::DXIL)?.data;
    let word = |at: usize| Some(u32::from_le_bytes(d.get(at..at + 4)?.try_into().ok()?) as usize);
    let (offset, size) = (word(16)?, word(20)?);
    d.get(8 + offset..8 + offset + size)
}

/// Runs the bitcode with root constants `inputs`: each dword of the UAV
/// the shader stored, by index.
fn execute(tools: &Tools, dxil: &[u8], inputs: [u32; 4]) -> Result<Vec<(usize, u32)>, String> {
    static RUNS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let n = RUNS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("victory-cpu-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let result = execute_in(tools, &dir, dxil, inputs);
    let _ = std::fs::remove_dir_all(&dir);
    result
}

fn execute_in(
    tools: &Tools,
    dir: &Path,
    dxil: &[u8],
    inputs: [u32; 4],
) -> Result<Vec<(usize, u32)>, String> {
    let file = |name: &str| dir.join(name);
    std::fs::write(file("shader.bc"), dxil).map_err(|e| e.to_string())?;
    let consts = inputs
        .iter()
        .chain(&[0; 4])
        .map(|x| format!("i32 {}", *x as i32))
        .collect::<Vec<_>>()
        .join(", ");
    let runtime = RUNTIME.replace(
        "@consts = global [8 x i32] zeroinitializer",
        &format!("@consts = global [8 x i32] [{consts}]"),
    );
    let runtime =
        format!("target datalayout = \"{LAYOUT}\"\ntarget triple = \"{TRIPLE}\"\n{runtime}");
    std::fs::write(file("runtime.ll"), runtime).map_err(|e| e.to_string())?;
    let run = |cmd: &mut Command| -> Result<String, String> {
        let out = cmd.output().map_err(|e| e.to_string())?;
        let text = String::from_utf8_lossy(&out.stdout).into_owned();
        if out.status.success() {
            Ok(text)
        } else {
            Err(format!(
                "{cmd:?}: {}{text}{}",
                out.status,
                String::from_utf8_lossy(&out.stderr)
            ))
        }
    };
    let host = run(Command::new(&tools.opt)
        .arg(format!("-data-layout={LAYOUT}"))
        .arg(format!("-mtriple={TRIPLE}"))
        .arg("-S")
        .arg(file("shader.bc"))
        .arg("-o")
        .arg("-"))?;
    std::fs::write(file("host.ll"), saturating_conversions(&host)).map_err(|e| e.to_string())?;
    run(Command::new(&tools.link)
        .arg(file("host.ll"))
        .arg(file("runtime.ll"))
        .arg("-o")
        .arg(file("run.bc")))?;
    let text = run(Command::new(&tools.lli)
        .arg("-entry-function=run")
        .arg(file("run.bc")))?;
    let mut stored = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let (written, value) = line
            .split_once(' ')
            .ok_or_else(|| format!("output: {text}"))?;
        if written == "1" {
            stored.push((
                i,
                u32::from_str_radix(value, 16).map_err(|e| e.to_string())?,
            ));
        }
    }
    Ok(stored)
}

/// `text` (an LLVM module) with each float-to-integer conversion made a
/// saturating one. In LLVM IR an out-of-range `fptosi`/`fptoui` is poison
/// (x86 gives 0x80000000); in DXIL it has D3D's meaning, which the
/// translation relies on: the result saturates, and NaN gives 0. That is
/// what `llvm.fpto[su]i.sat` computes.
fn saturating_conversions(text: &str) -> String {
    fn mangle(ty: &str) -> Option<&str> {
        match ty {
            "half" => Some("f16"),
            "float" => Some("f32"),
            "double" => Some("f64"),
            t if t.starts_with('i') => Some(t),
            _ => None,
        }
    }
    let mut out = String::with_capacity(text.len());
    let mut declared = std::collections::BTreeSet::new();
    for line in text.lines() {
        let rewritten = ["fptosi", "fptoui"].iter().find_map(|op| {
            let (lhs, rest) = line.split_once(&format!(" = {op} "))?;
            // `<from> <value> to <into>`, perhaps with metadata after.
            let (from, rest) = rest.split_once(' ')?;
            let (value, rest) = rest.split_once(" to ")?;
            let into = rest.split([',', ' ']).next()?;
            let name = format!("llvm.{op}.sat.{}.{}", mangle(into)?, mangle(from)?);
            declared.insert(format!("declare {into} @{name}({from})"));
            Some(format!("{lhs} = call {into} @{name}({from} {value})"))
        });
        out.push_str(rewritten.as_deref().unwrap_or(line));
        out.push('\n');
    }
    for d in declared {
        out.push_str(&d);
        out.push('\n');
    }
    out
}

/// `body` translated and run once per input set; `check(case, inputs, v0)`.
/// Says it was skipped when the tools are missing.
fn run_cases(body: &[u32], cases: &[[u32; 4]], mut check: impl FnMut(usize, [u32; 4], u32)) {
    let tools = match tools() {
        Ok(t) => t,
        Err(why) => return super::test_skip::skip(&format!("{why}; CPU run skipped")),
    };
    let dxil = translate(body).unwrap_or_else(|e| panic!("translation failed: {e}"));
    for (i, &inputs) in cases.iter().enumerate() {
        let stored = execute(tools, &dxil, inputs).unwrap_or_else(|e| panic!("case {i}: {e}"));
        let [(0, v0)] = stored[..] else {
            panic!("case {i}: the shader stored {stored:x?}, not one dword at 0");
        };
        check(i, inputs, v0);
    }
}

/// [`run_cases`] where each case is `[inputs, want]`.
fn expect(what: &str, body: &[u32], cases: &[([u32; 4], u32)]) {
    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(body, &inputs, |i, inp, got| {
        let want = cases[i].1;
        assert_eq!(
            got, want,
            "{what} case {i} {inp:08x?}: got {got:#010x}, want {want:#010x}"
        );
    });
}

fn f(x: f32) -> u32 {
    x.to_bits()
}

#[test]
fn bitcode_is_found_in_the_container() {
    // A real container (through the whole translation) and two broken ones.
    let dxil = translate(&[vop2(0x03, 0, vgpr(1), 2)]).expect("translates");
    assert_eq!(&dxil[..4], b"BC\xC0\xDE");
    assert!(bitcode(b"DXBC").is_none());
    assert!(bitcode(&[]).is_none());
}

#[test]
fn float_to_integer_conversions_become_saturating() {
    let text =
        "  %5 = fptosi float %4 to i32\n  %6 = fptoui float %4 to i32, !dbg !1\n  ret void\n";
    assert_eq!(
        saturating_conversions(text),
        "  %5 = call i32 @llvm.fptosi.sat.i32.f32(float %4)\n\
         \x20 %6 = call i32 @llvm.fptoui.sat.i32.f32(float %4)\n\
         \x20 ret void\n\
         declare i32 @llvm.fptosi.sat.i32.f32(float)\n\
         declare i32 @llvm.fptoui.sat.i32.f32(float)\n"
    );
}

#[test]
fn float_add_and_multiply_compute_what_they_should() {
    expect(
        "v_add_f32",
        &[vop2(0x03, 0, vgpr(1), 2)],
        &[
            ([f(1.5), f(2.25), 0, 0], f(3.75)),
            ([f(-0.0), f(0.0), 0, 0], f(0.0)),
            ([f(f32::INFINITY), f(1.0), 0, 0], f(f32::INFINITY)),
        ],
    );
    expect(
        "v_mul_f32",
        &[vop2(0x08, 0, vgpr(1), 2)],
        &[
            ([f(1.5), f(-4.0), 0, 0], f(-6.0)),
            ([f(3.0), f(0.5), 0, 0], f(1.5)),
        ],
    );
}

#[test]
fn integer_operations_compute_what_they_should() {
    expect(
        "v_add_nc_u32",
        &[vop2(0x25, 0, vgpr(1), 2)],
        &[
            ([0xFFFF_FFFF, 2, 0, 0], 1),
            ([0x7FFF_FFFF, 1, 0, 0], 0x8000_0000),
        ],
    );
    expect(
        "v_lshlrev_b32",
        &[vop2(0x1A, 0, vgpr(1), 2)],
        &[
            // Only the shift's low five bits count.
            ([33, 1, 0, 0], 2),
            ([4, 0x1234_5678, 0, 0], 0x2345_6780),
        ],
    );
    expect(
        "v_ashrrev_i32",
        &[vop2(0x18, 0, vgpr(1), 2)],
        &[([4, 0x8000_0000, 0, 0], 0xF800_0000)],
    );
    expect(
        "v_mul_hi_u32",
        &vop3(0x16A, 0, V123, 0, false),
        &[
            ([0xFFFF_FFFF, 0xFFFF_FFFF, 0, 0], 0xFFFF_FFFE),
            ([0x1_0000, 0x1_0000, 0, 0], 1),
        ],
    );
    expect(
        "v_mul_hi_i32",
        &vop3(0x16C, 0, V123, 0, false),
        &[
            ([0xFFFF_FFFF, 0xFFFF_FFFF, 0, 0], 0),
            ([0x8000_0000, 2, 0, 0], 0xFFFF_FFFF),
        ],
    );
    expect(
        "v_bfe_u32",
        &vop3(0x148, 0, V123, 0, false),
        &[([0xABCD_1234, 8, 8, 0], 0x12)],
    );
    expect(
        "v_bfe_i32",
        &vop3(0x149, 0, V123, 0, false),
        &[([0xABCD_1234, 28, 4, 0], 0xFFFF_FFFA)],
    );
    expect(
        "v_med3_i32",
        &vop3(0x158, 0, V123, 0, false),
        &[
            ([5, 0xFFFF_FFFF, 3, 0], 3),
            ([0x8000_0000, 0x7FFF_FFFF, 0, 0], 0),
        ],
    );
    expect(
        "v_bfrev_b32",
        &[vop1(0x38, 0, vgpr(1))],
        &[([1, 0, 0, 0], 0x8000_0000)],
    );
}

#[test]
fn bit_scans_count_from_the_ends_rdna2_names() {
    // v_ffbh_u32 counts from the top bit, v_ffbl_b32 from the bottom, and
    // v_ffbh_i32 skips the copies of the sign bit; no bit gives -1.
    expect(
        "v_ffbh_u32",
        &[vop1(0x39, 0, vgpr(1))],
        &[
            ([0x0001_0000, 0, 0, 0], 15),
            ([0, 0, 0, 0], u32::MAX),
            ([0x8000_0000, 0, 0, 0], 0),
        ],
    );
    expect(
        "v_ffbl_b32",
        &[vop1(0x3A, 0, vgpr(1))],
        &[([0x0001_0000, 0, 0, 0], 16), ([0, 0, 0, 0], u32::MAX)],
    );
    expect(
        "v_ffbh_i32",
        &[vop1(0x3B, 0, vgpr(1))],
        &[
            ([0xFFFF_0000, 0, 0, 0], 16),
            ([0x0000_FFFF, 0, 0, 0], 16),
            ([u32::MAX, 0, 0, 0], u32::MAX),
            ([0, 0, 0, 0], u32::MAX),
        ],
    );
}

#[test]
fn float_conversions_round_as_rdna2_does() {
    expect(
        "v_cvt_i32_f32",
        &[vop1(0x08, 0, vgpr(1))],
        &[
            // Toward zero, saturating; NaN gives 0.
            ([f(-2.75), 0, 0, 0], 0xFFFF_FFFE),
            ([f(3e9), 0, 0, 0], 0x7FFF_FFFF),
            ([f(-3e9), 0, 0, 0], 0x8000_0000),
            ([f(f32::NAN), 0, 0, 0], 0),
        ],
    );
    expect(
        "v_cvt_u32_f32",
        &[vop1(0x07, 0, vgpr(1))],
        &[
            ([f(-1.0), 0, 0, 0], 0),
            ([f(5e9), 0, 0, 0], u32::MAX),
            ([f(7.9), 0, 0, 0], 7),
        ],
    );
    expect(
        "v_cvt_f32_ubyte1",
        &[vop1(0x12, 0, vgpr(1))],
        &[([0x0000_C800, 0, 0, 0], f(200.0))],
    );
    expect(
        "v_rndne_f32",
        &[vop1(0x23, 0, vgpr(1))],
        &[([f(2.5), 0, 0, 0], f(2.0)), ([f(3.5), 0, 0, 0], f(4.0))],
    );
}

#[test]
fn med3_of_floats_with_nan_matches_rdna2() {
    // valu_tests.rs' hardware cases: with one NaN, RDNA2 gives the smaller
    // of the other two; with two, the one left.
    let nan = f(f32::NAN);
    expect(
        "v_med3_f32",
        &vop3(0x157, 0, V123, 0, false),
        &[
            ([f(1.0), f(2.0), f(3.0), 0], f(2.0)),
            ([f(3.0), f(1.0), f(2.0), 0], f(2.0)),
            ([f(1.0), f(2.0), nan, 0], f(1.0)),
            ([nan, f(1.0), f(0.0), 0], f(0.0)),
            ([nan, f(0.0), f(1.0), 0], f(0.0)),
            ([f(2.0), nan, f(1.0), 0], f(1.0)),
            ([f(-3.0), nan, f(4.0), 0], f(-3.0)),
            ([nan, f(5.0), nan, 0], f(5.0)),
            ([nan, nan, f(-2.0), 0], f(-2.0)),
            ([f(7.0), nan, nan, 0], f(7.0)),
        ],
    );
}

#[test]
fn packed_dot_products_match_rdna2() {
    // The same cases as valu_tests.rs' GPU run (and #44's v_dot8_i32_i4).
    expect(
        "v_dot8_i32_i4",
        &vop3p(0x18, 0, V123, false),
        &[
            ([0xFFFF_FFFF, 0x7777_7777, 100, 0], 44),
            ([0x8888_8888, 0x8888_8888, 0, 0], 512),
        ],
    );
    expect(
        "v_dot8_u32_u4",
        &vop3p(0x19, 0, V123, false),
        &[([0xFFFF_FFFF, 0xFFFF_FFFF, 1, 0], 1801)],
    );
    expect(
        "v_dot4_u32_u8",
        &vop3p(0x17, 0, V123, false),
        &[([0x0102_0304, 0x0101_0101, 5, 0], 15)],
    );
    // Clamped: the sum saturates instead of wrapping.
    expect(
        "v_dot4_i32_i8 clamp",
        &vop3p(0x16, 0, V123, true),
        &[
            ([0x7F7F_7F7F, 0x7F7F_7F7F, 0x7FFF_FFFF, 0], 0x7FFF_FFFF),
            ([0x8080_8080, 0x7F7F_7F7F, 0x8000_0000, 0], 0x8000_0000),
        ],
    );
}

#[test]
fn half_floats_round_to_nearest_even() {
    // v_cvt_f16_f32: 1 + 2^-11 is halfway between two f16 values and rounds
    // to the even one, 1.0; 1 + 3 * 2^-11 rounds up.
    expect(
        "v_cvt_f16_f32",
        &[vop1(0x0A, 0, vgpr(1))],
        &[
            ([f(1.0 + 1.0 / 2048.0), 0, 0, 0], 0x3C00),
            ([f(1.0 + 3.0 / 2048.0), 0, 0, 0], 0x3C02),
            ([f(65520.0), 0, 0, 0], 0x7C00),
        ],
    );
    expect(
        "v_cvt_f32_f16",
        &[vop1(0x0B, 0, vgpr(1))],
        &[([0xC500, 0, 0, 0], f(-5.0))],
    );
    // v_add_f16 writes the low half and keeps the destination's high half
    // (v0 is 0 here).
    expect(
        "v_add_f16",
        &[vop2(0x32, 0, vgpr(1), 2)],
        &[([0x3C00, 0x4000, 0, 0], 0x4200)],
    );
}

#[test]
fn neg_and_clamp_modifiers_apply() {
    // v_add_f32 through VOP3 (opcode 0x103) with neg on src1, then clamped.
    expect(
        "v_add_f32 -src1",
        &vop3(0x103, 0, V123, 0b010, false),
        &[([f(1.0), f(3.0), 0, 0], f(-2.0))],
    );
    expect(
        "v_add_f32 clamp",
        &vop3(0x103, 0, V123, 0, true),
        &[
            ([f(1.0), f(3.0), 0, 0], f(1.0)),
            ([f(-1.0), f(0.5), 0, 0], f(0.0)),
        ],
    );
}
