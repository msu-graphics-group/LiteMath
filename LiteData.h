#ifndef LITE_DATA_H
#define LITE_DATA_H

#if defined(__ANDROID__)
#include <android/asset_manager_jni.h>
#include <android/asset_manager.h>
#endif

namespace LiteData {

#if defined(__ANDROID__)

  void SetAssetManager(AAssetManager* assetManager);
  AAssetManager* GetAssetManager();
  void SetFileDir(const char* fileDir);
  const char* GetFileDir();

#endif

  char* ReadFile(const char* filepath, long &size);
  void WriteFile(const char* filepath, unsigned int size, const char* data);

} // namespace LiteData

#endif