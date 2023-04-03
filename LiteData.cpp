#include "LiteData.h"

#include <iostream>
#include <fstream>
#include <string.h>

namespace LiteData {

#if defined(__ANDROID__)
  static AAssetManager* g_AssetManager = nullptr;
  static char * g_fileDir = nullptr;

  void SetAssetManager(AAssetManager* assetManager) {
    g_AssetManager = assetManager;
  }

  AAssetManager* GetAssetManager() {
    return g_AssetManager;
  }

  void SetFileDir(const char* fileDir) {
    g_fileDir = const_cast<char*>(fileDir);
  }

  const char* GetFileDir() {
    return g_fileDir;
  }

  char * ReadFile(const char * filepath, long &size) {
    if(!g_AssetManager) {
      size = 0;
      return nullptr;
    }

    AAsset* file = AAssetManager_open(g_AssetManager, filepath, AASSET_MODE_BUFFER);
    size = AAsset_getLength(file);

    char* buffer = new char[size];

    AAsset_read(file, buffer, size);

    AAsset_close(file);
    return buffer;
  }

  void WriteFile(const char * filepath, const unsigned int size, const char * data) {
    if(!g_fileDir) {
      return;
    }

    char absFilepath[512];
    strcpy(absFilepath, g_fileDir);
    strcat(absFilepath, "/");
    strcat(absFilepath, filepath);

    std::ofstream outfile(absFilepath, std::ifstream::binary);

    if(!outfile) {
      return;
    }

    outfile.write(data, size);

    outfile.close();
  }

#else

  char * ReadFile(const char* filepath, long &size) {
    std::ifstream infile(filepath, std::ifstream::binary);

    if(!infile) {
      size = 0L;
      return nullptr;
    }

    infile.seekg(0, infile.end);
    size = infile.tellg();
    infile.seekg(0);

    char* buffer = new char[size];
    infile.read(buffer, size);

    infile.close();
    return buffer;
  }

  void WriteFile(const char* filepath, const unsigned int size, const char* data) {
    std::ofstream outfile(filepath, std::ifstream::binary);

    if(!outfile) {
      return;
    }

    outfile.write(data, size);

    outfile.close();
  }

#endif

} // namespace LiteData