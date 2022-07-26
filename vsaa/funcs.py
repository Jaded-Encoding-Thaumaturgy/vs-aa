import vapoursynth as vs
from vskernels import Catrom, Kernel
from vsutil import get_w, join, split

from .abstract import SingleRater, SuperSampler
from .antialiasers import Eedi3SR, Nnedi3SS
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
