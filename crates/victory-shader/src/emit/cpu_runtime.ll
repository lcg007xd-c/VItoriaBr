; Stands in for the `dx.op` intrinsics a translated compute shader calls, so
; LLVM's `lli` can run it on the CPU (see `cpu_tests.rs`). The test fills
; @consts (row 0: s0..s3; row 1: the descriptor table, slot 0) and reads
; what the shader stored back from @run's output: one line per dword of the
; UAV, "<written> <value>".
;
; Each intrinsic follows DXIL.rst: the opcode is the first argument, and the
; float ones keep IEEE semantics (no flush to zero, NaN in, NaN out unless
; the operation says otherwise).

%dx.types.Handle = type { ptr }
%dx.types.ResBind = type { i32, i32, i32, i8 }
%dx.types.ResourceProperties = type { i32, i32 }
%dx.types.CBufRet.i32 = type { i32, i32, i32, i32 }
%dx.types.Dimensions = type { i32, i32, i32, i32 }

@consts = global [8 x i32] zeroinitializer
@out = global [16 x i32] zeroinitializer
@written = global [16 x i8] zeroinitializer
@fmt = private constant [9 x i8] c"%u %08x\0A\00"

declare i32 @printf(ptr, ...)
declare void @main()

declare float @llvm.fabs.f32(float)
declare float @llvm.cos.f32(float)
declare float @llvm.sin.f32(float)
declare float @llvm.exp2.f32(float)
declare float @llvm.log2.f32(float)
declare float @llvm.sqrt.f32(float)
declare float @llvm.roundeven.f32(float)
declare float @llvm.floor.f32(float)
declare float @llvm.ceil.f32(float)
declare float @llvm.trunc.f32(float)
declare float @llvm.maxnum.f32(float, float)
declare float @llvm.minnum.f32(float, float)
declare float @llvm.fma.f32(float, float, float)
declare i32 @llvm.bitreverse.i32(i32)
declare i32 @llvm.ctpop.i32(i32)
declare i32 @llvm.cttz.i32(i32, i1)
declare i32 @llvm.ctlz.i32(i32, i1)
declare i32 @llvm.smax.i32(i32, i32)
declare i32 @llvm.smin.i32(i32, i32)
declare i32 @llvm.umax.i32(i32, i32)
declare i32 @llvm.umin.i32(i32, i32)

define i32 @run() {
entry:
  call void @main()
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %n, %loop ]
  %p = getelementptr [16 x i32], ptr @out, i32 0, i32 %i
  %v = load i32, ptr %p
  %q = getelementptr [16 x i8], ptr @written, i32 0, i32 %i
  %w = load i8, ptr %q
  %wz = zext i8 %w to i32
  %x = call i32 (ptr, ...) @printf(ptr @fmt, i32 %wz, i32 %v)
  %n = add i32 %i, 1
  %done = icmp eq i32 %n, 16
  br i1 %done, label %end, label %loop
end:
  ret i32 0
}

; Resources: the one CBV is @consts, every heap view is the UAV @out.
define %dx.types.Handle @dx.op.createHandleFromBinding(i32 %op, %dx.types.ResBind %b, i32 %i, i1 %nu) {
  ret %dx.types.Handle { ptr @consts }
}
define %dx.types.Handle @dx.op.createHandleFromHeap(i32 %op, i32 %i, i1 %s, i1 %nu) {
  ret %dx.types.Handle { ptr @out }
}
define %dx.types.Handle @dx.op.annotateHandle(i32 %op, %dx.types.Handle %h, %dx.types.ResourceProperties %p) {
  ret %dx.types.Handle %h
}
define %dx.types.Dimensions @dx.op.getDimensions(i32 %op, %dx.types.Handle %h, i32 %mip) {
  ret %dx.types.Dimensions { i32 64, i32 0, i32 0, i32 0 }
}
define %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 %op, %dx.types.Handle %h, i32 %row) {
  %p = extractvalue %dx.types.Handle %h, 0
  %i = mul i32 %row, 4
  %ok = icmp ult i32 %row, 2
  %j = select i1 %ok, i32 %i, i32 0
  %q = getelementptr i32, ptr %p, i32 %j
  %v = load <4 x i32>, ptr %q, align 4
  %a = extractelement <4 x i32> %v, i32 0
  %b = extractelement <4 x i32> %v, i32 1
  %c = extractelement <4 x i32> %v, i32 2
  %d = extractelement <4 x i32> %v, i32 3
  %r0 = insertvalue %dx.types.CBufRet.i32 undef, i32 %a, 0
  %r1 = insertvalue %dx.types.CBufRet.i32 %r0, i32 %b, 1
  %r2 = insertvalue %dx.types.CBufRet.i32 %r1, i32 %c, 2
  %r3 = insertvalue %dx.types.CBufRet.i32 %r2, i32 %d, 3
  ret %dx.types.CBufRet.i32 %r3
}

; A raw buffer's index is a byte offset; a dword past the 64 bytes is dropped.
define void @store_dword(i32 %at, i32 %v) {
  %ok = icmp ult i32 %at, 16
  br i1 %ok, label %st, label %done
st:
  %p = getelementptr [16 x i32], ptr @out, i32 0, i32 %at
  store i32 %v, ptr %p
  %q = getelementptr [16 x i8], ptr @written, i32 0, i32 %at
  store i8 1, ptr %q
  br label %done
done:
  ret void
}
define void @store_lane(i32 %at, i32 %k, i32 %v, i8 %mask) {
  %kk = trunc i32 %k to i8
  %m = shl i8 1, %kk
  %on = and i8 %mask, %m
  %yes = icmp ne i8 %on, 0
  br i1 %yes, label %st, label %done
st:
  %d = add i32 %at, %k
  call void @store_dword(i32 %d, i32 %v)
  br label %done
done:
  ret void
}
define void @dx.op.rawBufferStore.i32(i32 %op, %dx.types.Handle %h, i32 %index, i32 %off, i32 %v0, i32 %v1, i32 %v2, i32 %v3, i8 %mask, i32 %align) {
  %at = lshr i32 %index, 2
  call void @store_lane(i32 %at, i32 0, i32 %v0, i8 %mask)
  call void @store_lane(i32 %at, i32 1, i32 %v1, i8 %mask)
  call void @store_lane(i32 %at, i32 2, i32 %v2, i8 %mask)
  call void @store_lane(i32 %at, i32 3, i32 %v3, i8 %mask)
  ret void
}

; One thread in one group.
define i32 @dx.op.threadIdInGroup.i32(i32 %op, i32 %c) {
  ret i32 0
}
define i32 @dx.op.threadId.i32(i32 %op, i32 %c) {
  ret i32 0
}
define i32 @dx.op.groupId.i32(i32 %op, i32 %c) {
  ret i32 0
}
define i32 @dx.op.flattenedThreadIdInGroup.i32(i32 %op) {
  ret i32 0
}
define i32 @dx.op.waveGetLaneIndex(i32 %op) {
  ret i32 0
}
define i32 @dx.op.waveGetLaneCount(i32 %op) {
  ret i32 32
}
define i32 @dx.op.waveReadLaneFirst.i32(i32 %op, i32 %x) {
  ret i32 %x
}
define void @dx.op.barrier(i32 %op, i32 %flags) {
  ret void
}

; Float conversions to and from f16 bits (DXIL rounds to nearest even).
define i32 @dx.op.legacyF32ToF16(i32 %op, float %x) {
  %h = fptrunc float %x to half
  %b = bitcast half %h to i16
  %r = zext i16 %b to i32
  ret i32 %r
}
define float @dx.op.legacyF16ToF32(i32 %op, i32 %x) {
  %t = trunc i32 %x to i16
  %h = bitcast i16 %t to half
  %r = fpext half %h to float
  ret float %r
}

define float @dx.op.unary.f32(i32 %op, float %x) {
  switch i32 %op, label %bad [
    i32 6, label %fabs
    i32 7, label %saturate
    i32 12, label %cos
    i32 13, label %sin
    i32 21, label %exp
    i32 22, label %frc
    i32 23, label %log
    i32 24, label %sqrt
    i32 25, label %rsqrt
    i32 26, label %round_ne
    i32 27, label %round_ni
    i32 28, label %round_pi
    i32 29, label %round_z
  ]
fabs:
  %a = call float @llvm.fabs.f32(float %x)
  ret float %a
saturate:
  ; NaN gives 0.
  %s0 = call float @llvm.maxnum.f32(float %x, float 0.0)
  %s1 = call float @llvm.minnum.f32(float %s0, float 1.0)
  ret float %s1
cos:
  %c = call float @llvm.cos.f32(float %x)
  ret float %c
sin:
  %si = call float @llvm.sin.f32(float %x)
  ret float %si
exp:
  %e = call float @llvm.exp2.f32(float %x)
  ret float %e
frc:
  %fl = call float @llvm.floor.f32(float %x)
  %f = fsub float %x, %fl
  ret float %f
log:
  %l = call float @llvm.log2.f32(float %x)
  ret float %l
sqrt:
  %q = call float @llvm.sqrt.f32(float %x)
  ret float %q
rsqrt:
  %rq = call float @llvm.sqrt.f32(float %x)
  %r = fdiv float 1.0, %rq
  ret float %r
round_ne:
  %ne = call float @llvm.roundeven.f32(float %x)
  ret float %ne
round_ni:
  %ni = call float @llvm.floor.f32(float %x)
  ret float %ni
round_pi:
  %pi = call float @llvm.ceil.f32(float %x)
  ret float %pi
round_z:
  %z = call float @llvm.trunc.f32(float %x)
  ret float %z
bad:
  call void @unknown(i32 %op)
  unreachable
}

define float @dx.op.binary.f32(i32 %op, float %a, float %b) {
  switch i32 %op, label %bad [
    i32 35, label %max
    i32 36, label %min
  ]
max:
  %x = call float @llvm.maxnum.f32(float %a, float %b)
  ret float %x
min:
  %n = call float @llvm.minnum.f32(float %a, float %b)
  ret float %n
bad:
  call void @unknown(i32 %op)
  unreachable
}

define float @dx.op.tertiary.f32(i32 %op, float %a, float %b, float %c) {
  switch i32 %op, label %bad [
    i32 46, label %fmad
    i32 47, label %fma
  ]
fmad:
  ; Not fused: the product rounds, then the sum. (A driver may fuse FMad,
  ; so a translation that needs one rounding cannot be checked here.)
  %p = fmul float %a, %b
  %s = fadd float %p, %c
  ret float %s
fma:
  %f = call float @llvm.fma.f32(float %a, float %b, float %c)
  ret float %f
bad:
  call void @unknown(i32 %op)
  unreachable
}

define i32 @dx.op.unary.i32(i32 %op, i32 %x) {
  switch i32 %op, label %bad [
    i32 30, label %bfrev
  ]
bfrev:
  %r = call i32 @llvm.bitreverse.i32(i32 %x)
  ret i32 %r
bad:
  call void @unknown(i32 %op)
  unreachable
}

; Countbits; FirstbitLo; FirstbitHi and FirstbitSHi count from the top bit
; (DXC subtracts from 31 for HLSL's firstbithigh); -1 when no bit is found.
define i32 @dx.op.unaryBits.i32(i32 %op, i32 %x) {
  %neg = icmp slt i32 %x, 0
  %inv = xor i32 %x, -1
  %sx = select i1 %neg, i32 %inv, i32 %x
  switch i32 %op, label %bad [
    i32 31, label %count
    i32 32, label %lo
    i32 33, label %hi
    i32 34, label %shi
  ]
count:
  %c = call i32 @llvm.ctpop.i32(i32 %x)
  ret i32 %c
lo:
  %z0 = icmp eq i32 %x, 0
  %t = call i32 @llvm.cttz.i32(i32 %x, i1 false)
  %l = select i1 %z0, i32 -1, i32 %t
  ret i32 %l
hi:
  %z1 = icmp eq i32 %x, 0
  %h = call i32 @llvm.ctlz.i32(i32 %x, i1 false)
  %hr = select i1 %z1, i32 -1, i32 %h
  ret i32 %hr
shi:
  %z2 = icmp eq i32 %sx, 0
  %sh = call i32 @llvm.ctlz.i32(i32 %sx, i1 false)
  %sr = select i1 %z2, i32 -1, i32 %sh
  ret i32 %sr
bad:
  call void @unknown(i32 %op)
  unreachable
}

define i32 @dx.op.binary.i32(i32 %op, i32 %a, i32 %b) {
  switch i32 %op, label %bad [
    i32 37, label %imax
    i32 38, label %imin
    i32 39, label %umax
    i32 40, label %umin
  ]
imax:
  %r0 = call i32 @llvm.smax.i32(i32 %a, i32 %b)
  ret i32 %r0
imin:
  %r1 = call i32 @llvm.smin.i32(i32 %a, i32 %b)
  ret i32 %r1
umax:
  %r2 = call i32 @llvm.umax.i32(i32 %a, i32 %b)
  ret i32 %r2
umin:
  %r3 = call i32 @llvm.umin.i32(i32 %a, i32 %b)
  ret i32 %r3
bad:
  call void @unknown(i32 %op)
  unreachable
}

; Dot4AddI8Packed (163) and Dot4AddU8Packed (164): acc + the four byte products.
define i32 @dx.op.dot4AddPacked.i32(i32 %op, i32 %acc, i32 %a, i32 %b) {
  %va = bitcast i32 %a to <4 x i8>
  %vb = bitcast i32 %b to <4 x i8>
  %signed = icmp eq i32 %op, 163
  %sa = sext <4 x i8> %va to <4 x i32>
  %sb = sext <4 x i8> %vb to <4 x i32>
  %ua = zext <4 x i8> %va to <4 x i32>
  %ub = zext <4 x i8> %vb to <4 x i32>
  %xa = select i1 %signed, <4 x i32> %sa, <4 x i32> %ua
  %xb = select i1 %signed, <4 x i32> %sb, <4 x i32> %ub
  %p = mul <4 x i32> %xa, %xb
  %p0 = extractelement <4 x i32> %p, i32 0
  %p1 = extractelement <4 x i32> %p, i32 1
  %p2 = extractelement <4 x i32> %p, i32 2
  %p3 = extractelement <4 x i32> %p, i32 3
  %s0 = add i32 %acc, %p0
  %s1 = add i32 %s0, %p1
  %s2 = add i32 %s1, %p2
  %s3 = add i32 %s2, %p3
  ret i32 %s3
}

; An opcode this runtime does not know: stop, and say which.
@bad = private constant [30 x i8] c"cpu runtime: no dx.op %u\0A\00\00\00\00\00"
declare void @exit(i32)
define void @unknown(i32 %op) {
  %x = call i32 (ptr, ...) @printf(ptr @bad, i32 %op)
  call void @exit(i32 3)
  ret void
}
