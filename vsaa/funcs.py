from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vsexprtools import norm_expr_planes
from vskernels import Catrom, Scaler, ScalerT, Spline144
from vsmasktools import EdgeDetect, EdgeDetectT, Prewitt, ScharrTCanny
from vsrgtools import RepairMode, box_blur, contrasharpening_median, median_clips, repair, unsharp_masked
from vstools import (
    MISSING, CustomOverflowError, CustomRuntimeError, MissingT, PlanesT, check_ref_clip, check_variable, core,
    get_depth, get_peak_value, get_w, get_y, join, normalize_planes, scale_8bit, scale_value, split, vs
)

from .abstract import Antialiaser, SingleRater
from .antialiasers import Eedi3, Eedi3SR, Nnedi3SR, Nnedi3SS, Znedi3
from .enums import AADirection
from .mask import resize_aa_mask

__all__ = [
    'pre_aa',
    'upscaled_sraa',
    'transpose_aa',
    'clamp_aa', 'masked_clamp_aa',
    'fine_aa',
    'based_aa'
]


def pre_aa(
    clip: vs.VideoNode, radius: int = 1, strength: int = 100,
    aa: type[Antialiaser] | Antialiaser = Znedi3(0, pscrn=1),
    **kwargs: Any
) -> vs.VideoNode:
    if isinstance(aa, Antialiaser):
        aa = aa.copy(field=3, **kwargs)
    else:
        aa = aa(field=3, **kwargs)

    for _ in range(2):
        bob = aa.interpolate(clip, False)
        sharp = unsharp_masked(clip, radius, strength)
        limit = median_clips(sharp, clip, bob[::2], bob[1::2])
        clip = limit.std.Transpose()

    return clip


def upscaled_sraa(
    clip: vs.VideoNode, rfactor: float = 1.5,
    width: int | None = None, height: int | None = None,
    ssfunc: ScalerT = Nnedi3SS(), aafunc: SingleRater = Eedi3SR(),
    direction: AADirection = AADirection.BOTH,
    downscaler: ScalerT = Catrom(), planes: PlanesT = 0
) -> vs.VideoNode:
    """
    Super-sampled single-rate AA for heavy aliasing and broken line-art.

    It works by super-sampling the clip, performing AA, and then downscaling again.
    Downscaling can be disabled by setting `downscaler` to `None`, returning the super-sampled luma clip.
    The dimensions of the downscaled clip can also be adjusted by setting `height` or `width`.
    Setting either `height`, `width` or 1,2 in planes will also scale the chroma accordingly.

    :param clip:            Clip to process.
    :param rfactor:         Image enlargement factor.
                            It is not recommended to go below 1.3
    :param width:           Target resolution width. If None, determined from `height`.
    :param height:          Target resolution height.
    :param ssfunc:          Super-sampler used for upscaling before AA.
    :param aafunc:          Function used to antialias after super-sampling.
    :param direction:       Direction in which antialiasing will be performed.
    :param downscaler:      Downscaler to use after super-sampling.
    :param planes:          Planes to do antialiasing on.

    :return:                Antialiased clip.

    :raises ValueError:     ``rfactor`` is not above 1.
    """
    assert clip.format

    planes = normalize_planes(clip, planes)

    if height is None:
        height = clip.height

    if width is None:
        if height == clip.height:
            width = clip.width
        else:
            width = get_w(height, clip)

    aa_chroma = 1 in planes or 2 in planes
    scale_chroma = ((clip.width, clip.height) != (width, height)) or aa_chroma

    work_clip, *chroma = [clip] if scale_chroma else split(clip)

    if rfactor <= 1:
        raise ValueError('upscaled_sraa: rfactor must be above 1!')

    ssw = (round(work_clip.width * rfactor) + 1) & ~1
    ssh = (round(work_clip.height * rfactor) + 1) & ~1

    ssfunc = Scaler.ensure_obj(ssfunc, upscaled_sraa)
    downscaler = Scaler.ensure_obj(downscaler, upscaled_sraa)

    up = ssfunc.scale(work_clip, ssw, ssh)
    up, *chroma = [up] if aa_chroma else (split(up) if scale_chroma else [up, *chroma])

    aa = aafunc.aa(up, *direction.to_yx())

    if downscaler:
        aa = downscaler.scale(aa, width, height)

    if not chroma:
        return aa

    return join([aa, *chroma], clip.format.color_family)


def transpose_aa(clip: vs.VideoNode, aafunc: SingleRater) -> vs.VideoNode:
    """
    Perform transposed AA.

    :param clip:        Clip to process.
    :param aafun:       Antialiasing function.
    :return:            Antialiased clip.
    """
    assert clip.format

    work_clip, *chroma = split(clip)

    aafunc.transpose_first = True
    aafunc.drop_fields = False

    aa_y = aafunc.aa(work_clip, AADirection.BOTH)

    if not chroma:
        return aa_y

    return join([aa_y, *chroma], clip.format.color_family)


def clamp_aa(
    src: vs.VideoNode, weak: vs.VideoNode, strong: vs.VideoNode,
    strength: float = 1, planes: PlanesT = 0
) -> vs.VideoNode:
    """
    Clamp stronger AAs to weaker AAs.
    Useful for clamping :py:func:`upscaled_sraa` or :py:func:`Eedi3SR`
    to :py:func:`Nnedi3SR` for a strong but more precise AA.

    :param src:         Non-AA'd source clip.
    :param weak:        Weakly-AA'd clip.
    :param strong:      Strongly-AA'd clip.
    :param strength:    Clamping strength.
    :param planes:      Planes to process.

    :return:            Clip with clamped anti-aliasing.
    """
    assert src.format

    planes = normalize_planes(src, planes)

    if src.format.sample_type == vs.INTEGER:
        thr = strength * get_peak_value(src)
    else:
        thr = strength / 219

    if thr == 0:
        return median_clips(src, weak, strong, planes=planes)

    expr = f'x y - XYD! XYD@ x z - XZD! XZD@ xor x XYD@ abs XZD@ abs < z y {thr} + min y {thr} - max z ? ?'

    return core.akarin.Expr([src, weak, strong], norm_expr_planes(src, expr, planes))


def masked_clamp_aa(
    clip: vs.VideoNode, strength: float = 1,
    mthr: float = 0.25, mask: vs.VideoNode | EdgeDetectT | None = None,
    weak_aa: SingleRater | None = None, strong_aa: SingleRater | None = None,
    opencl: bool | None = True
) -> vs.VideoNode:
    """
    Clamp a strong aa to a weaker one for the purpose of reducing the stronger's artifacts.

    :param clip:                Clip to process.
    :param strength:            Set threshold strength for over/underflow value for clamping.
    :param mthr:                Binarize threshold for the mask, float.
    :param mask:                Clip to use for custom mask or an EdgeDetect to use custom masker.
    :param weak_aa:             SingleRater for the weaker aa.
    :param strong_aa:           SingleRater for the stronger aa.
    :param opencl:              Wheter to force OpenCL acceleration, None to leave as is.

    :return:                    Antialiased clip.
    """
    assert clip.format

    work_clip, *chroma = split(clip)

    if not isinstance(mask, vs.VideoNode):
        bin_thr = scale_value(mthr, 32, get_depth(clip))

        mask = ScharrTCanny.ensure_obj(mask).edgemask(work_clip)  # type: ignore
        mask = mask.std.Binarize(bin_thr)
        mask = mask.std.Maximum()
        mask = box_blur(mask)
        mask = mask.std.Minimum().std.Deflate()

    if weak_aa is None:
        weak_aa = Nnedi3SR(
            opencl=hasattr(core, 'nnedi3cl') if opencl is None else opencl
        )
    elif opencl is not None and hasattr(weak_aa, 'opencl'):
        weak_aa._opencl = opencl  # type: ignore[attr-defined]

    if strong_aa is None:
        strong_aa = Eedi3SR(opencl=opencl is None or opencl)
    elif opencl is not None and hasattr(strong_aa, 'opencl'):
        strong_aa._opencl = opencl  # type: ignore[attr-defined]

    weak = transpose_aa(work_clip, weak_aa)
    strong = transpose_aa(work_clip, strong_aa)

    clamped = clamp_aa(work_clip, weak, strong, strength)

    merged = work_clip.std.MaskedMerge(clamped, mask)  # type: ignore

    if not chroma:
        return merged

    return join([merged, *chroma], clip.format.color_family)


def fine_aa(
    clip: vs.VideoNode, taa: bool = False,
    singlerater: SingleRater = Eedi3SR(),
    rep: int | RepairMode = RepairMode.LINE_CLIP_STRONG
) -> vs.VideoNode:
    """
    Taa and optionally repair clip that results in overall lighter anti-aliasing, downscaled with Spline kernel.

    :param clip:            Clip to process.
    :param taa:             Whether to perform a normal or transposed aa
    :param singlerater:     Singlerater used for aa.
    :param rep:             Repair mode.

    :return:                Antialiased clip.
    """
    assert clip.format

    singlerater.shifter = Spline144()

    work_clip, *chroma = split(clip)

    if taa:
        aa = transpose_aa(work_clip, singlerater)
    else:
        aa = singlerater.aa(work_clip, AADirection.BOTH)

    contra = contrasharpening_median(work_clip, aa)

    repaired = repair(contra, work_clip, rep)

    if not chroma:
        return repaired

    return join([repaired, *chroma], clip.format.color_family)


if TYPE_CHECKING:
    from vsdenoise import Prefilter
    from vsscale import SSIM, FSRCNNXShader, FSRCNNXShaderT, ShaderFile

    def based_aa(
        clip: vs.VideoNode, rfactor: float = 2.0,
        mask_thr: int = 60, lmask: vs.VideoNode | EdgeDetectT = Prewitt,
        downscaler: ScalerT = SSIM,
        supersampler: ScalerT | FSRCNNXShaderT | ShaderFile | Path = FSRCNNXShader.x56,
        antialiaser: Antialiaser = Eedi3(0.125, 0.25, vthresh0=12, vthresh1=24, field=1, sclip_aa=None),
        prefilter: Prefilter | vs.VideoNode = Prefilter.NONE, show_mask: bool = False, **kwargs: Any
    ) -> vs.VideoNode:
        ...
else:
    def based_aa(
        clip: vs.VideoNode, rfactor: float = 2.0,
        mask_thr: int = 60, lmask: vs.VideoNode | EdgeDetectT = Prewitt,
        downscaler: ScalerT | MissingT = MISSING,
        supersampler: ScalerT | FSRCNNXShaderT | ShaderFile | Path | MissingT = MISSING,
        antialiaser: Antialiaser = Eedi3(0.125, 0.25, vthresh0=12, vthresh1=24, field=1, sclip_aa=None),
        prefilter: Prefilter | vs.VideoNode | MissingT = MISSING, show_mask: bool = False, **kwargs: Any
    ) -> vs.VideoNode:
        try:
            from vsscale import SSIM, FSRCNNXShader, PlaceboShader  # noqa: F811
        except ModuleNotFoundError:
            raise CustomRuntimeError(
                'You\'re missing the "vsscale" package! You can install it with "pip install vsscale".', based_aa
            )

        try:
            from vsdenoise import Prefilter  # noqa: F811
        except ModuleNotFoundError:
            raise CustomRuntimeError(
                'You\'re missing the "vsdenoise" package! You can install it with "pip install vsdenoise".', based_aa
            )

        if downscaler is MISSING:
            downscaler = SSIM

        if supersampler is MISSING:
            supersampler = FSRCNNXShader.x56

        if prefilter is MISSING:
            prefilter = Prefilter.NONE

        assert check_variable(clip, based_aa)

        aaw = (round(clip.width * rfactor) + 1) & ~1
        aah = (round(clip.height * rfactor) + 1) & ~1

        clip_y = get_y(clip)
        if isinstance(prefilter, vs.VideoNode):
            wclip_y = get_y(prefilter)
            check_ref_clip(clip_y, wclip_y, based_aa)
        else:
            wclip_y = prefilter(clip_y, **kwargs)

        if rfactor < 1.0:
            raise CustomOverflowError(
                f'"rfactor" must be bigger than 1.0! ({rfactor})', based_aa
            )

        if not isinstance(lmask, vs.VideoNode):
            lmask = EdgeDetect.ensure_obj(lmask, based_aa)

            if mask_thr > 255:
                raise CustomOverflowError(
                    f'"mask_thr" must be less or equal than 255! ({mask_thr})', based_aa
                )

            lmask = lmask.edgemask(wclip_y).std.Binarize(scale_8bit(wclip_y, mask_thr))
            lmask = box_blur(lmask.std.Maximum()).std.Limiter()

        if show_mask:
            return lmask

        if isinstance(supersampler, (str, Path)):
            supersampler = PlaceboShader(supersampler)

        supersampler = Scaler.ensure_obj(supersampler, based_aa)
        downscaler = Scaler.ensure_obj(downscaler, based_aa)

        mclip_up = resize_aa_mask(lmask, aaw, aah)

        aa = clip_y.std.Transpose()
        aa = supersampler.scale(aa, aa.width * ceil(rfactor), aa.height * ceil(rfactor))
        aa = waa = downscaler.scale(aa, aah, aaw)

        if clip_y is not wclip_y:
            waa = wclip_y.std.Transpose()
            waa = supersampler.scale(waa, waa.width * ceil(rfactor), waa.height * ceil(rfactor))
            waa = downscaler.scale(waa, aah, aaw)

        def _aa_mclip(clip: vs.VideoNode, wclip: vs.VideoNode, mclip: vs.VideoNode) -> vs.VideoNode:
            return core.std.Expr([
                clip, antialiaser.interpolate(wclip, False, sclip=wclip), mclip
            ], 'z y x ?')

        aa = _aa_mclip(aa, waa, mclip_up.std.Transpose())

        aa = _aa_mclip(aa.std.Transpose(), waa.std.Transpose(), mclip_up)

        aa = downscaler.scale(aa, clip_y.width, clip_y.height)

        aa_merge = clip_y.std.MaskedMerge(aa, lmask)

        if clip.format.num_planes == 1:
            return aa_merge

        return join(aa_merge, clip, clip.format.color_family)
