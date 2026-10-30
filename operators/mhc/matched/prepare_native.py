"""Generate an isolated AscendC shifted-Pre operator from the pinned upstream source."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

REV = 'ba5162411c919e15fbabb28298e8b420c11ad905'


def once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError('Pinned source context mismatch: ' + old[:100])
    return text.replace(old, new)


def function_body(text, signature, body):
    start = text.index('{', text.index(signature))
    depth = 1
    end = start + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[:start] + '{\n' + body.strip('\n') + '\n}' + text[end:]


def transform(relative, text):
    text = text.replace('mhc_pre', 'mhc_shifted_pre').replace('MhcPre', 'MhcShiftedPre')
    text = text.replace('MHC_PRE', 'MHC_SHIFTED_PRE')
    if relative == 'op_host/mhc_pre_def.cpp':
        text = once(text, 'this->Input("gamma")\n            .ParamType(OPTIONAL)',
                    'this->Input("prev_pre")\n            .ParamType(REQUIRED)')
        text = once(text, 'this->Output("h_pre")\n            .ParamType(OPTIONAL)',
                    'this->Output("h_pre")\n            .ParamType(REQUIRED)')
        text = text.replace('        this->AICore().AddConfig("ascend910b");', '')
        text = text.replace('        this->AICore().AddConfig("ascend910_93");', '')
        text = text.replace('        this->AICore().AddConfig("ascend350", aicConfig);', '')
    elif relative == 'op_host/op_api/aclnn_mhc_pre.cpp':
        # Retain the argument position but give it the explicit shifted semantics.
        text = text.replace('gammaOptional', 'prevPre')
        text = function_body(text, 'static aclnnStatus CheckParams(', r'''
    CHECK_RET(op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckNotNull(params), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(params.prevPre != nullptr && params.hRes != nullptr && params.hPreOptional != nullptr,
              ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(params.invRmsOptional == nullptr && params.hMixOptional == nullptr, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTensorRank(params.x, "x", 3), ACLNN_ERR_PARAM_INVALID);
    auto t = params.x->GetViewShape().GetDim(0);
    CHECK_RET(t > 0 && CheckTensorShape(params.x, {t, 4, 5120}, "x"), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTensorDtype(params.x, "x", DataType::DT_BF16), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTensorShape(params.phi, {24, 20480}, "phi") &&
              CheckTensorShape(params.alpha, {3}, "alpha") &&
              CheckTensorShape(params.bias, {24}, "bias") &&
              CheckTensorShape(params.prevPre, {t, 4}, "prev_pre") &&
              CheckTensorShape(params.hIn, {t, 5120}, "hIn") &&
              CheckTensorShape(params.hPost, {t, 4}, "hPost") &&
              CheckTensorShape(params.hRes, {t, 4, 4}, "hRes") &&
              CheckTensorShape(params.hPreOptional, {t, 4}, "next_pre"), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckInputDtype(params) && CheckOutputDtype(params) &&
              CheckTensorDtype(params.hPreOptional, "next_pre", DataType::DT_FLOAT), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormat(params) && CheckTensorFormat(params.hPreOptional, "next_pre"), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckOpImplMode(params), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
''')
        text = function_body(text, 'static aclnnStatus CopyOptionalOutputs(', r'''
    (void)invRmsOut;
    (void)hMixOut;
    auto ret = l0op::ViewCopy(hPreOut, params.hPreOptional, executor);
    CHECK_RET(ret != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
''')
        # Former generic/A2 validation helpers remain for a reviewable upstream diff.
        text = text.replace('static bool ', '[[maybe_unused]] static bool ')
        text = text.replace('static int64_t Factorial(', '[[maybe_unused]] static int64_t Factorial(')
    elif relative == 'op_host/op_tiling/arch35/mhc_pre_tiling.cpp':
        text = once(text, 'hasGamma_ = (gammaTensor == nullptr) ? 0 : 1;',
                    '// Input slot 4 carries prev_pre, never an RMS gamma.\n    '
                    'OP_CHECK_NULL_WITH_CONTEXT(context_, gammaTensor);\n    hasGamma_ = 0;')
    elif relative == 'op_kernel/arch35/mhc_pre_vector_compute.h':
        text = once(text, '''        if (outFlag_) {
            uint64_t offset = globalOffsetM_ + offsetT;
            DataCopyOutHPre(offset, totalElem);
        }''', '''        // next_pre is required even though backward caches are disabled.
        uint64_t offset = globalOffsetM_ + offsetT;
        DataCopyOutHPre(offset, totalElem);''')
        text = once(text, '''        AIV1ProcessHPre(offsetT, lenT);
        AIV1ProcessHInBatched(offsetT, lenT, xBufferBytes, outBufferBytes);''', '''        AIV1ProcessHPre(offsetT, lenT);
        // Reuse the pre UB only after its next_pre DMA has completed.
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID2);
        DataCopyExtParams prevCopy;
        prevCopy.blockCount = 1U;
        prevCopy.blockLen = static_cast<uint32_t>(lenT * N_ * sizeof(P));
        prevCopy.srcStride = 0U;
        prevCopy.dstStride = 0U;
        DataCopyPadExtParams<P> prevPad{false, 0, 0, 0};
        // gammaGm_ is the inherited field for input slot 4 (prev_pre in this operator).
        DataCopyPad(hPreBuff_, gammaGm_[(globalOffsetM_ + offsetT) * N_], prevCopy, prevPad);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
        AIV1ProcessHInBatched(offsetT, lenT, xBufferBytes, outBufferBytes);''')
    return text


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--record', type=Path, required=True)
    a = p.parse_args()
    revision = subprocess.check_output(['git', '-C', str(a.source), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != REV:
        raise RuntimeError('Unexpected source revision: ' + revision)
    src = a.source / 'mhc/mhc_pre'
    dst = a.source / 'mhc/mhc_shifted_pre'
    plans = []
    for file in sorted(src.rglob('*')):
        if not file.is_file():
            continue
        rel = file.relative_to(src).as_posix()
        if rel.split('/')[0] not in ('op_host', 'op_kernel', 'CMakeLists.txt'):
            continue
        base = subprocess.check_output(['git', '-C', str(a.source), 'show', REV + ':mhc/mhc_pre/' + rel]).decode()
        if file.read_text(encoding='utf-8') != base:
            raise RuntimeError('Modified upstream input: ' + rel)
        new_rel = rel.replace('mhc_pre', 'mhc_shifted_pre')
        result = transform(rel, base)
        target = dst / new_rel
        if target.exists() and target.read_text(encoding='utf-8') != result:
            raise RuntimeError('Generated file contains other changes: ' + str(target))
        plans.append((rel, new_rel, base, result, target))
    if not plans:
        raise RuntimeError('No upstream sources found')
    allowed = {target.resolve() for _, _, _, _, target in plans}
    if dst.exists() and any(f.resolve() not in allowed for f in dst.rglob('*') if f.is_file()):
        raise RuntimeError('Unexpected files in generated operator; preserve them before regenerating')
    a.record.mkdir(parents=True, exist_ok=True)
    hashes = {}
    diffs = []
    for rel, new_rel, base, result, target in plans:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(result, encoding='utf-8', newline='\n')
        hashes[new_rel] = hashlib.sha256(result.encode()).hexdigest()
        diffs.extend(difflib.unified_diff(base.splitlines(True), result.splitlines(True),
                     fromfile='upstream/' + rel, tofile='mhc_shifted_pre/' + new_rel))
    (a.record / 'adaptation.diff').write_text(''.join(diffs), encoding='utf-8')
    (a.record / 'source.json').write_text(json.dumps(dict(revision=REV, operator='MhcShiftedPre',
        semantics='input4=prev_pre; required next_pre; no backward caches; native old-pre apply',
        files=hashes), indent=2), encoding='utf-8')
    print('Generated MhcShiftedPre:', dst, 'files=', len(plans))


if __name__ == '__main__':
    main()
