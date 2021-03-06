#include "transcoder/basisu_transcoder.h"

using namespace basist;

#ifdef __cplusplus
extern "C" {
#endif

static basisu_transcoder transcoder;
void init() { basisu_transcoder_init(); }

bool start(const void *pData, uint32_t data_size) {
  return transcoder.start_transcoding(pData, data_size);
}
uint32_t totalImages(const void *pData, uint32_t data_size) {
  return transcoder.get_total_images(pData, data_size);
}

bool imageInfo(const void *pData, uint32_t data_size,
               basisu_image_info image_info, uint32_t image_index) {
  return transcoder.get_image_info(pData, data_size, image_info, image_index);
}

bool imageLevelInfo(const void *pData, uint32_t data_size,
                    basisu_image_level_info &level_info, uint32_t image_index,
                    uint32_t level_index) {

  return transcoder.get_image_level_info(pData, data_size, level_info,
                                         image_index, level_index);
}

bool transcodeImageLevel(const void *pData, uint32_t data_size,
                         basisu_image_level_info &level_info,
                         void *pOutput_blocks, transcoder_texture_format fmt) {
  return transcoder.transcode_image_level(
      pData, data_size, level_info.m_image_index, level_info.m_level_index,
      pOutput_blocks, level_info.m_total_blocks, fmt);
}

uint32_t bytesPerBlock(transcoder_texture_format fmt) {
  return basis_get_bytes_per_block_or_pixel(fmt);
}

#ifdef __cplusplus
}
#endif
