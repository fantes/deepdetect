
// protoUtils.cc ---

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

#include "protoUtils.h"

#include <google/protobuf/io/coded_stream.h>                                                           
#include <google/protobuf/io/zero_copy_stream_impl.h>                                                  
#include <google/protobuf/text_format.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "src/caffe.pb.h"
#include <algorithm>
#include "mllibstrategy.h"


namespace dd
{

int findNClasses(const std::string source, bool bbox)
{
  caffe::NetParameter net;
  if (!TRTReadProtoFromTextFile(source.c_str(),&net))
    return -1;
  int nlayers = net.layer_size();
  if (bbox)
    {
      for (int i= nlayers-1; i>= 0; --i)
	{
	  caffe::LayerParameter lparam = net.layer(i);
	  if (lparam.type() == "DetectionOutput")
	    return lparam.detection_output_param().num_classes();
	}
    }
  for (int i= nlayers-1; i>= 0; --i)
    {
      caffe::LayerParameter lparam = net.layer(i);
      if (lparam.type() == "InnerProduct")
	return lparam.inner_product_param().num_output();
    }
  return -1;
}

int findTopK(const std::string source)
{
  caffe::NetParameter net;
  if (!TRTReadProtoFromTextFile(source.c_str(),&net))
    return -1;
  int nlayers = net.layer_size();
  for (int i= nlayers-1; i>= 0; --i)
    {
      caffe::LayerParameter lparam = net.layer(i);
      if (lparam.type() == "DetectionOutput")
        return lparam.detection_output_param().nms_param().top_k();
    }
  return -1;
}


int findAlphabetSize(const std::string source)
{
  caffe::NetParameter net;
  if (!TRTReadProtoFromTextFile(source.c_str(),&net))
    return -1;
  for (int i= net.layer_size()-1; i>= 0; --i)
    {
      caffe::LayerParameter lparam = net.layer(i);
      if (lparam.type() == "InnerProduct")
        return lparam.inner_product_param().num_output();
    }
  return -1;
}

int findTimeSteps(const std::string source)
{
  caffe::NetParameter net;
  if (!TRTReadProtoFromTextFile(source.c_str(),&net))
    return -1;
  for (int i= net.layer_size()-1; i>= 0; --i)
    {
      caffe::LayerParameter lparam = net.layer(i);
      if (lparam.type() == "ContinuationIndicator")
        return lparam.continuation_indicator_param().time_step();
    }
  return -1;
}



void visualizeNet(const nvinfer1::INetworkDefinition* network)
{
  std::cout << "NETWORK\ninputs: " << std::endl;
  for (int i=0; i< network->getNbInputs(); ++i)
    std::cout << "   " << network->getInput(i)->getName() << std::endl;
  for (int i=0; i< network->getNbLayers(); ++i)
    {
      nvinfer1::ILayer* l = network->getLayer(i);
      int nin = l->getNbInputs();
      int nout = l->getNbOutputs();
      std::string name(l->getName());
      std::cout << "layer " << i << " " << name
                << " i:" << nin  << " o:" << nout << std::endl;
      if (name.find("lstm")!=name.npos)
        {
          nin = 1;
          nout = 1;
        }
      for (int j=0; j< nin; ++j)
        std::cout << "     input " << j << " " << l->getInput(j)->getName() << std::endl;
      for (int j=0; j< nout; ++j)
        std::cout << "     output " << j << " " << l->getOutput(j)->getName() << std::endl;
    }

}


std::string firstLSTMInput(caffe::NetParameter &source_net)
{
  for (int i =0; i<source_net.layer_size(); ++i)
    {
      caffe::LayerParameter lparam = source_net.layer(i);
      if (lparam.type() == "LSTM")
        return lparam.bottom(0);
    }
  return std::string("");
}


int fixProto(const std::string dest, const std::string source, const std::string binary_proto)
{
  caffe::NetParameter source_net;
  caffe::NetParameter dest_net;

  caffe::NetParameter binary_net;
  if (!TRTReadProtoFromBinaryFile(binary_proto.c_str(),&binary_net))
    return 3;

  if (!TRTReadProtoFromTextFile(source.c_str(),&source_net))
    return 1;

  dest_net.set_name(source_net.name());

  for (int i =0; i<source_net.layer_size(); ++i)
    {
      caffe::LayerParameter lparam = source_net.layer(i);

      if (lparam.type() == "Permute")
        {
          if (lparam.top(0) == firstLSTMInput(source_net))
            {
              caffe::PermuteParameter* pp = lparam.mutable_permute_param();
              int oldo0 = pp->order(0);
              pp->set_order(0,pp->order(1));
              pp->set_order(1,oldo0);
            }
          caffe::LayerParameter* dlparam = dest_net.add_layer();
          *dlparam = lparam;
        }
      else if (lparam.type() == "MemoryData")
        {
          dest_net.add_input(lparam.top(0));
          caffe::BlobShape* is = dest_net.add_input_shape();
          is->add_dim(lparam.memory_data_param().batch_size());
          is->add_dim(lparam.memory_data_param().channels());
          is->add_dim(lparam.memory_data_param().height());
          is->add_dim(lparam.memory_data_param().width());
        }
      else if (lparam.type() == "Flatten")
        {
          caffe::LayerParameter* rparam = dest_net.add_layer();
          rparam->set_name(lparam.name());
          rparam->set_type("Reshape");
          rparam->add_bottom(lparam.bottom(0));
          rparam->add_top(lparam.top(0));
          int faxis = lparam.flatten_param().axis();
          caffe::ReshapeParameter * rp = rparam->mutable_reshape_param();
          caffe::BlobShape* bs = rp->mutable_shape();
          for (int i=0; i<faxis; ++i)
            bs->add_dim(0);
          bs->add_dim(-1);
          for (int i=faxis+1; i<4; ++i)
            bs->add_dim(1);
        }
      else if (lparam.type() == "DetectionOutput")
        {
          caffe::LayerParameter* dlparam = dest_net.add_layer();
          caffe::NonMaximumSuppressionParameter* nmsp =
            lparam.mutable_detection_output_param()->mutable_nms_param();
          nmsp->clear_soft_nms();
          nmsp->clear_theta();
          *dlparam = lparam;
          dlparam->add_top("keep_count");
        }
      else if (lparam.type() == "ContinuationIndicator")
        {
          // simply skip this layer
        }
      else if (lparam.type() == "InnerProduct")
        {
          caffe::LayerParameter* dlparam = dest_net.add_layer();
          *dlparam = lparam;
          dlparam->set_type("InnerProductWithAxis");
          dlparam->mutable_inner_product_param()->set_axis(dlparam->mutable_inner_product_param()->axis()-1);
        }
      else if (lparam.type() == "Softmax")
        {
          caffe::LayerParameter* dlparam = dest_net.add_layer();
          *dlparam = lparam;
          dlparam->mutable_softmax_param()->set_axis(dlparam->mutable_softmax_param()->axis()-1);
        }
      else
        {
          caffe::LayerParameter* dlparam = dest_net.add_layer();
          *dlparam = lparam;
        }
    }

  
  if (!TRTWriteProtoToTextFile(dest_net,dest.c_str()))
    return 2;
  return 0;
}



bool TRTReadProtoFromBinaryFile(const char* filename, google::protobuf::Message* proto) {
  int fd = open(filename, O_RDONLY);
  if (fd == -1)
    return false;
  google::protobuf::io::ZeroCopyInputStream* raw_input =
    new google::protobuf::io::FileInputStream(fd);
  google::protobuf::io::CodedInputStream* coded_input =
    new google::protobuf::io::CodedInputStream(raw_input);
  coded_input->SetTotalBytesLimit(INT_MAX, 536870912);

  bool success = proto->ParseFromCodedStream(coded_input);

  delete coded_input;
  delete raw_input;
  close(fd);
  return success;
}

bool TRTReadProtoFromTextFile(const char* filename, google::protobuf::Message* proto)
{
  int fd = open(filename, O_RDONLY);
  if (fd == -1)
    return false;
  google::protobuf::io::FileInputStream* input = new google::protobuf::io::FileInputStream(fd);
  bool success = google::protobuf::TextFormat::Parse(input, proto);
  delete input;
  close(fd);
  return success;
}

bool TRTWriteProtoToTextFile(const google::protobuf::Message& proto, const char* filename)
{
  int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd == -1)
    return false;
  google::protobuf::io::FileOutputStream* output = new google::protobuf::io::FileOutputStream(fd);
  bool success = google::protobuf::TextFormat::Print(proto, output);
  delete output;
  close(fd);
  return success;
}

}
