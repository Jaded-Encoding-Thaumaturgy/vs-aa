import vapoursynth as vs
from vsexprtools.util import PlanesT, norm_expr_planes, normalise_planes
from vskernels import Catrom, Kernel
from vsutil import get_peak_value, get_w, join, split

from .abstract import SingleRater, SuperSampler
from .antialiasers import Eedi3SR, Nnedi3SS
from vsrgtools import median_clips
from .enums import AADirection

core = vs.core


def upscaled_sraa(
    clip: vs.VideoNode, rfactor: float = 1.5,
    width: int | None = None, height: int | None = None,
    ssfunc: SuperSampler = Nnedi3SS(), aafunc: SingleRater = Eedi3SR(),
    direction: AADirection = AADirection.BOTH,
    downscaler: Kernel = Catrom()
) -> vs.VideoNode:
    """
    :param clip:            Clip to process, only luma will be processed.
    :param rfactor:         Image enlargement factor.
                            It is not recommended to go below 1.3
    :param width:           Target resolution width. If None, determined from `height`.
    :param height:          Target resolution height.
    :param ssfunc:          Super-sampler used for upscaling before AA.
    :param aafunc:          Downscaler to use after super-sampling.
    :param aafun:           Function used to antialias after super-sampling.

    :return:                Antialiased clip.

    :raises ValueError:     ``rfactor`` is not above 1.
    """
    assert clip.format

    work_clip, *chroma = split(clip)

    if rfactor <= 1:
        raise ValueError('upscaled_sraa: rfactor must be above 1!')

    ssw = (round(work_clip.width * rfactor) + 1) & ~1
    ssh = (round(work_clip.height * rfactor) + 1) & ~1

    if height is None:
        height = work_clip.height

    if width is None:
        if height == work_clip.height:
            width = work_clip.width
        else:
            width = get_w(height, aspect_ratio=clip.width / clip.height)

    up_y = ssfunc.scale(work_clip, ssw, ssh)

    aa_y = aafunc.aa(up_y, *direction.to_yx())

    if downscaler:
        aa_y = downscaler.scale(aa_y, width, height)
    elif not chroma or (clip.width, clip.height) != (width, height):
        return aa_y

    return join([aa_y, *chroma], clip.format.color_family)


def taa(clip: vs.VideoNode, aafunc: SingleRater) -> vs.VideoNode:
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
    Useful for clamping :py:func:`upscaled_sraa` or :py:func:`Eedi3SR` to :py:func:`Nnedi3SR`
    for a strong but more precise AA.

    Original function written by `Zastin <https://github.com/kgrabs>`_,
    modified by LightArrowsEXE, Setsugen no ao.

    :param src:         Non-AA'd source clip.
    :param weak:        Weakly-AA'd clip (eg: :py:func:`lvsfunc.aa.nnedi3`).
    :param strong:      Strongly-AA'd clip (eg: :py:func:`lvsfunc.aa.eedi3`).
    :param strength:    Clamping strength (Default: 1).

    :return:            Clip with clamped anti-aliasing.
    """
    assert src.format

    planes = normalise_planes(src, planes)

    if src.format.sample_type == vs.INTEGER:
        thr = strength * get_peak_value(src)
    else:
        thr = strength / 219

    if thr == 0:
        return median_clips([src, weak, strong], planes)

    expr = f'x y - XYD! XYD@ x z - XZD! XZD@ xor x XYD@ abs XZD@ abs < z y {thr} + min y {thr} - max z ? ?'

    return core.akarin.Expr([src, weak, strong], norm_expr_planes(src, expr, planes))

