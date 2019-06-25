
// protoUtils.h ---

// Copyright (C) 2019 Jolibrain http://www.jolibrain.com

// Author: Guillaume Infantes <guillaume.infantes@jolibrain.com>

// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 3
// of the License, or (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.


#ifndef PROTO_UTILS_H
#define PROTO_UTILS_H


#include <google/protobuf/io/coded_stream.h>                                                            
#include <google/protobuf/io/zero_copy_stream_impl.h>                                                   
#include <google/protobuf/text_format.h>
#include <spdlog/spdlog.h>
#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "src/caffe.pb.h"

namespace dd
{
  int fixProto(const std::string dest, const std::string source, const std::string binary_proto);
  int findNClasses(const std::string source, bool bbox);
  int findTopK(const std::string source);
  int findTimeSteps(const std::string source);
  int findAlphabetSize(const std::string source);
  std::string firstLSTMInput(caffe::NetParameter &source_net);
  bool TRTReadProtoFromBinaryFile(const char* filename, google::protobuf::Message* proto);
  bool TRTReadProtoFromTextFile(const char* filename, google::protobuf::Message* proto);
  bool TRTWriteProtoToTextFile(const google::protobuf::Message& proto, const char* filename);
  nvinfer1::ILayer* findLayerByName(const nvinfer1::INetworkDefinition* network, const std::string lname);

  void visualizeNet(const nvinfer1::INetworkDefinition* network);


}
#endif
