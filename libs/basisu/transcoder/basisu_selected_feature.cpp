#define BASISU_DEVEL_MESSAGES 1
#define BASISU_FORCE_DEVEL_MESSAGES	1

// Disable ktx2 and zstd
#define BASISD_SUPPORT_KTX2 0
#define BASISD_SUPPORT_KTX2_ZSTD 0

// select compress feature
#define BASISD_SUPPORT_UASTC 0
#define BASISD_SUPPORT_DXT1 0
#define BASISD_SUPPORT_DXT5A 0
#define BASISD_SUPPORT_BC7 0
#define BASISD_SUPPORT_BC7_MODE5 1
#define BASISD_SUPPORT_PVRTC1 0
#define BASISD_SUPPORT_ETC2_EAC_A8 0
#define BASISD_SUPPORT_ASTC 0
#define BASISD_SUPPORT_ATC 0
#define BASISD_SUPPORT_ASTC_HIGHER_OPAQUE_QUALITY 0
#define BASISD_SUPPORT_ETC2_EAC_RG11 0
#define BASISD_SUPPORT_FXT1 0
#define BASISD_SUPPORT_PVRTC2 0

#include "basisu_transcoder.cpp"