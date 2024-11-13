from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from vsexprtools import complexpr_available, norm_expr
from vskernels import Bilinear, Box, Catrom, NoScale, Scaler, ScalerT
from vsmasktools import EdgeDetect, EdgeDetectT, Prewitt, ScharrTCanny
from vsrgtools import MeanMode, bilateral, box_blur, unsharp_masked
from vstools import (
    MISSING, CustomRuntimeError, CustomValueError, FormatsMismatchError, FunctionUtil, KwargsT, MissingT,
    PlanesT, VSFunction, get_peak_value, get_y, plane, scale_mask, vs
)

from .abstract import Antialiaser
from .antialiasers import Eedi3, Nnedi3

__all__ = [
    'pre_aa',
    'clamp_aa',
    'based_aa'
]


class _pre_aa:
    def custom(
        self, clip: vs.VideoNode, sharpen: VSFunction,
        aa: type[Antialiaser] | Antialiaser = Nnedi3,
        planes: PlanesT = None, **kwargs: Any
    ) -> vs.VideoNode:
        func = FunctionUtil(clip, pre_aa, planes)

        field = kwargs.pop('field', 3)
        if field < 2:
            field += 2

        if isinstance(aa, Antialiaser):
            aa = aa.copy(field=field, **kwargs)  # type: ignore
        else:
            aa = aa(field=field, **kwargs)

        wclip = func.work_clip

        for _ in range(2):
            bob = aa.interpolate(wclip, False)
            sharp = sharpen(wclip)
            limit = MeanMode.MEDIAN(sharp, wclip, bob[::2], bob[1::2])
            wclip = limit.std.Transpose()

        return func.return_clip(wclip)

    def __call__(
        self, clip: vs.VideoNode, radius: int = 1, strength: int = 100,
        aa: type[Antialiaser] | Antialiaser = Nnedi3,
        planes: PlanesT = None, **kwargs: Any
    ) -> vs.VideoNode:
        return self.custom(
            clip, partial(unsharp_masked, radius=radius, strength=strength), aa, planes, **kwargs
        )


pre_aa = _pre_aa()


def clamp_aa(
    clip: vs.VideoNode, strength: float = 1.0,
    mthr: float = 0.25, mask: vs.VideoNode | EdgeDetectT | None = None,
    weak_aa: vs.VideoNode | Antialiaser = Nnedi3(),
    strong_aa: vs.VideoNode | Antialiaser = Eedi3(),
    opencl: bool | None = False, ref: vs.VideoNode | None = None,
    planes: PlanesT = 0
) -> vs.VideoNode:
    """
    Clamp a strong aa to a weaker one for the purpose of reducing the stronger's artifacts.

    :param clip:                Clip to process.
    :param strength:            Set threshold strength for over/underflow value for clamping.
    :param mthr:                Binarize threshold for the mask, float.
    :param mask:                Clip to use for custom mask or an EdgeDetect to use custom masker.
    :param weak_aa:             Antialiaser for the weaker aa.
    :param strong_aa:           Antialiaser for the stronger aa.
    :param opencl:              Whether to force OpenCL acceleration, None to leave as is.
    :param ref:                 Reference clip for clamping.

    :return:                    Antialiased clip.
    """

    func = FunctionUtil(clip, clamp_aa, planes, (vs.YUV, vs.GRAY))

    if not isinstance(weak_aa, vs.VideoNode):
        if opencl is not None and hasattr(weak_aa, 'opencl'):
            weak_aa.opencl = opencl

        weak_aa = weak_aa.aa(func.work_clip)
    elif func.luma_only:
        weak_aa = get_y(weak_aa)

    if not isinstance(strong_aa, vs.VideoNode):
        if opencl is not None and hasattr(strong_aa, 'opencl'):
            strong_aa.opencl = opencl

        strong_aa = strong_aa.aa(func.work_clip)
    elif func.luma_only:
        strong_aa = get_y(strong_aa)

    if ref and func.luma_only:
        ref = get_y(ref)

    if func.work_clip.format.sample_type == vs.INTEGER:
        thr = strength * get_peak_value(func.work_clip)
    else:
        thr = strength / 219

    if thr == 0:
        clamped = MeanMode.MEDIAN(func.work_clip, weak_aa, strong_aa, planes=planes)
    else:
        if ref:
            if complexpr_available:
                expr = f'y z - ZD! y a - AD! ZD@ AD@ xor x ZD@ abs AD@ abs < a z {thr} + min z {thr} - max a ? ?'
            else:
                expr = f'y z - y a - xor x y z - abs y a - abs < a z {thr} + min z {thr} - max a ? ?'
        else:
            if complexpr_available:
                expr = f'x y - XYD! x z - XZD! XYD@ XZD@ xor x XYD@ abs XZD@ abs < z y {thr} + min y {thr} - max z ? ?'
            else:
                expr = f'x y - x z - xor x x y - abs x z - abs < z y {thr} + min y {thr} - max z ? ?'

            clamped = norm_expr(
                [func.work_clip, ref, weak_aa, strong_aa] if ref else [func.work_clip, strong_aa, strong_aa],
                expr, func.norm_planes
            )

    if mask:
        if not isinstance(mask, vs.VideoNode):
            bin_thr = scale_mask(mthr, 32, clip)

            mask = ScharrTCanny.ensure_obj(mask).edgemask(func.work_clip)  # type: ignore
            mask = box_blur(mask.std.Binarize(bin_thr).std.Maximum())
            mask = mask.std.Minimum().std.Deflate()

        clamped = func.work_clip.std.MaskedMerge(clamped, mask, func.norm_planes)  # type: ignore

    return func.return_clip(clamped)


if TYPE_CHECKING:
    from vsscale import ArtCNN, ShaderFile

    def based_aa(
        clip: vs.VideoNode, rfactor: float = 2.0,
        mask: vs.VideoNode | EdgeDetectT | Literal[False] = Prewitt, mask_thr: int = 60, pskip: bool = True,
        downscaler: ScalerT | None = None,
        supersampler: ScalerT | ShaderFile | Path | Literal[False] = ArtCNN.C16F64,
        double_rate: bool = False,
        eedi3_kwargs: KwargsT | None = dict(alpha=0.125, beta=0.25, vthresh0=12, vthresh1=24, field=1),
        prefilter: vs.VideoNode | VSFunction | None = None, postfilter: VSFunction | None | Literal[False] = None,
        show_mask: bool = False, planes: PlanesT = 0,
        **kwargs: Any
    ) -> vs.VideoNode:
        """
        Perform based anti-aliasing on a video clip.

        This function works by super- or downsampling the clip and applying Eedi3 to that image.
        The result is then merged with the original clip using an edge mask, and it's limited
        to areas where Eedi3 was actually applied.

        Sharp supersamplers will yield better results, so long as they do not introduce too much ringing.
        For downscalers, you will want to use a neutral kernel.

        :param clip:            Clip to process.
        :param rfactor:         Resize factor for supersampling. Values above 1.0 are recommended.
                                Lower values may be useful for particularly extremely aliased content.
                                Values closer to 1.0 will perform faster at the cost of precision.
                                This value must be greater than 0.0. Default: 2.0.
        :param mask:            Edge detection mask or function to generate it.  Default: Prewitt.
        :param mask_thr:        Threshold for edge detection mask. Must be less than or equal to 255.
                                Only used if an EdgeDetect class is passed to `mask`. Default: 60.
        :param pskip:           Whether to skip processing if EEDI3 had no contribution to the pixel's output.
        :param downscaler:      Scaler used for downscaling after anti-aliasing. This should ideally be
                                a relatively sharp kernel that doesn't introduce too much haloing.
                                If None, downscaler will be set to Box if the scale factor is an integer
                                (after rounding), and Catrom otherwise.
                                If rfactor is below 1.0, the downscaler will be used before antialiasing instead,
                                and the supersampler will be used to scale the clip back to its original resolution.
                                Default: None.
        :param supersampler:    Scaler used for supersampling before anti-aliasing. If False, no supersampling
                                is performed. If rfactor is below 1.0, the downscaler will be used before
                                antialiasing instead, and the supersampler will be used to scale the clip
                                back to its original resolution.
                                The supersampler should ideally be fairly sharp without
                                introducing too much ringing.
                                Default: ArtCNN.C16F64.
        :param double_rate:     Whether to use double-rate antialiasing.
                                If True, both fields will be processed separately, which may improve
                                anti-aliasing strength at the cost of increased processing time and detail loss.
                                Default: False.
        :param eedi3_kwargs:    Keyword arguments to pass on to EEDI3.
        :param prefilter:       Prefilter to apply before anti-aliasing.
                                Must be a VideoNode, a function that takes a VideoNode and returns a VideoNode,
                                or None. Default: None.
        :param postfilter:      Postfilter to apply after anti-aliasing.
                                Must be a function that takes a VideoNode and returns a VideoNode, or None.
                                If None, applies a median-filtered bilateral smoother to clean halos
                                created during antialiasing. Default: None.
        :param show_mask:       If True, returns the edge detection mask instead of the processed clip.
                                Default: False
        :param planes:          Planes to process. Default: Luma only.

        :return:                Anti-aliased clip or edge detection mask if show_mask is True.

        :raises CustomRuntimeError:     If required packages are missing.
        :raises CustomValueError:       If rfactor is not above 0.0, or invalid prefilter/postfilter is passed.
        """

        ...
else:
    def based_aa(
        clip: vs.VideoNode, rfactor: float = 2.0,
        mask: vs.VideoNode | EdgeDetectT | Literal[False] = Prewitt, mask_thr: int = 60, pskip: bool = True,
        downscaler: ScalerT | None = None,
        supersampler: ScalerT | ShaderFile | Path | Literal[False] | MissingT = MISSING,
        double_rate: bool = False,
        eedi3_kwargs: KwargsT | None = dict(alpha=0.125, beta=0.25, vthresh0=12, vthresh1=24, field=1),
        prefilter: vs.VideoNode | VSFunction | None = None, postfilter: VSFunction | None | Literal[False] = None,
        show_mask: bool = False, planes: PlanesT = 0,
        **kwargs: Any
    ) -> vs.VideoNode:

        func = FunctionUtil(clip, based_aa, planes, (vs.YUV, vs.GRAY))

        if supersampler is False:
            supersampler = downscaler = NoScale
        else:
            if supersampler is MISSING:
                try:
                    from vsscale import ArtCNN  # noqa: F811
                except ModuleNotFoundError:
                    raise CustomRuntimeError(
                        'You\'re missing the "vsscale" package! Install it with "pip install vsscale".', based_aa
                    )

                supersampler = ArtCNN.C16F64()
            elif isinstance(supersampler, (str, Path)):
                try:
                    from vsscale import PlaceboShader  # noqa: F811
                except ModuleNotFoundError:
                    raise CustomRuntimeError(
                        'You\'re missing the "vsscale" package! Install it with "pip install vsscale".', based_aa
                    )

                supersampler = PlaceboShader(supersampler)

        if rfactor <= 0.0:
            raise CustomValueError('rfactor must be greater than 0!', based_aa, rfactor)

        aaw, aah = [(round(r * rfactor) + 1) & ~1 for r in (func.work_clip.width, func.work_clip.height)]

        if downscaler is None:
            downscaler = Box if (
                max(aaw, func.work_clip.width) % min(aaw, func.work_clip.width) == 0
                and max(aah, func.work_clip.height) % min(aah, func.work_clip.height) == 0
            ) else Catrom

        if rfactor < 1.0:
            downscaler, supersampler = supersampler, downscaler

        if mask and not isinstance(mask, vs.VideoNode):
            mask_thr = scale_mask(min(mask_thr, 255), 8, func.work_clip)
            mask = EdgeDetect.ensure_obj(mask, based_aa)
            mask = mask.edgemask(plane(func.work_clip, 0))
            mask = mask.std.Binarize(mask_thr)
            mask = box_blur(mask.std.Maximum()).std.Limiter()

        if show_mask:
            return mask

        if prefilter:
            if isinstance(prefilter, vs.VideoNode):
                FormatsMismatchError.check(based_aa, func.work_clip, prefilter)
                ss_clip = prefilter
            elif callable(prefilter):
                ss_clip = prefilter(func.work_clip)
            else:
                raise CustomValueError('Invalid prefilter!', based_aa, prefilter)
        else:
            ss_clip = func.work_clip

        supersampler = Scaler.ensure_obj(supersampler, based_aa)
        downscaler = Scaler.ensure_obj(downscaler, based_aa)

        ss = supersampler.scale(ss_clip, aaw, aah)
        mclip = Bilinear.scale(mask, ss.width, ss.height) if mask else None

        if double_rate:
            aa = Eedi3(mclip=mclip, sclip_aa=True).draa(ss, **eedi3_kwargs | kwargs)
        else:
            aa = Eedi3(mclip=mclip, sclip_aa=True).aa(ss, **eedi3_kwargs | kwargs)

        aa = downscaler.scale(aa, func.work_clip.width, func.work_clip.height)

        if postfilter is None:
            aa_out = MeanMode.MEDIAN(aa, func.work_clip, bilateral(aa))
        elif callable(postfilter):
            aa_out = postfilter(aa)
        elif postfilter is False:
            aa_out = aa
        else:
            raise CustomValueError('Invalid postfilter!', based_aa, postfilter)

        if pskip:
            no_aa = downscaler.scale(ss, func.work_clip.width, func.work_clip.height)
            aa_out = norm_expr([func.work_clip, aa_out, aa, no_aa], "z a = x y ?")

        if mask:
            aa_out = func.work_clip.std.MaskedMerge(aa_out, mask)

        return func.return_clip(aa_out)
