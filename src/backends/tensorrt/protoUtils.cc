
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

using google::protobuf::io::FileInputStream;                                                            
using google::protobuf::io::FileOutputStream;                                                           
using google::protobuf::io::ZeroCopyInputStream;                                                        
using google::protobuf::io::CodedInputStream;                                                           
using google::protobuf::io::ZeroCopyOutputStream;                                                       
using google::protobuf::io::CodedOutputStream;                                                          



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
  int nlayers = net.layer_size();
  for (int i= nlayers-1; i>= 0; --i)
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
  int nlayers = net.layer_size();
  for (int i= nlayers-1; i>= 0; --i)
    {
      caffe::LayerParameter lparam = net.layer(i);
      if (lparam.type() == "ContinuationIndicator")
        return lparam.continuation_indicator_param().time_step();
    }
  return -1;
}


nvinfer::ILayer* findLayerByName(const nvinfer1::INetworkDefinition* network, const std::string lname)
{
  for (int i =0; i< network->getNbLayers(); ++i)
    {
      nvinfer1::ILayer * l = network->getlayer(i);
      if (l->getName() == lname)
	return l;
    }
  return nullptr;
}

  
int addUnparsablesFromProto(nvinfer1::INetworkDefinition* network, const std::string source,
			    const nvcaffeparser1::IBlobNameToTensor* b2t,
			    spdlog::logger* logger)
{
  caffe::NetParameter source_net;
  std::map<std::string, nvinfer1::ITensor*> t2t;
  if (!TRTReadProtoFromTextFile(source.c_str(),&source_net))
    return 1;
  bool first_lstm = true;
  int seq_size = -1;
  nvinfer1::ITensor * inputTensor = nullptr;
  for (int i =0; i<nlayers; ++i)
    {
      caffe::LayerParameter lparam = source_net.layer(i);
      if (lparam.type() == "ContinuationIndicator")
	{
	  seq_size = lparam.continuation_indicator_param().time_step();
	}
      else if (lparam.type() == "LSTM")
	{
	  if (first_lstm) // add a permute layer to permute first 2 axes (T,N)
	    {
	      std::string input = lparam.bottom(0);
	      inputTensor = b2t->find(input.c_str());
	      // permute layer is done through shuffle layer in tensorRT !!!?!
	      IShuffleLayer * sl = network->addShuffle(*inputTensor);
	      nvinfer1::Permutation p;
	      p.order[0] = 1;
	      p.order[1] = 0;
	      p.order[2] = 2;
	      sl->setFirstTranspose(p);
	      sl->setReshapeDimensions(Dims3{0,0,0});
	      inputTensor = sl->getOutput(0);
	    }
	  int num_out = lparam.recurrent_param().num_output();
	  auto rnn = network->addRNNv2(*inputTensor, 1, num_out, seq_size, RNNOperation::kLSTM);
	  std::string out = laparam.top(0);
	  inputTensor = rnn->getOutput(0);
	  t2t.insert(std::pair<std::string,nvinfer1::ITensor*>(out, inputTensor));
	}
      else
	{
	  for (int b = 0; b< lparam.bottom_size(); ++b)
	    {
	      std::map<std::string, ITensor*>::iterator it = t2t.find(lparam.bottom(b));
	      if (it != std::map::end)
		{
		  // first find layer in already translated ones
		  nvinfer::ILayer * atl = findLayerByName(network, lparam.name());
		  if (atl == nullptr)
		    {
		      logger->error("could not find layer {} for replacing input {} with output of manulaly created layer", lparam.name(), lparam.bottom(b));
		      throw MLLibInternalException("fatal error while creating network from caffe parser + manually adding layers");
		    }
		  // then update its input with ouput from manually inserter layer
		  atl->setInput(b,it->second);
		}
	    }
	}
    }
}


  
int fixProto(const std::string dest, const std::string source)
{
  caffe::NetParameter source_net;
  caffe::NetParameter dest_net;
  if (!TRTReadProtoFromTextFile(source.c_str(),&source_net))
    return 1;

  dest_net.set_name(source_net.name());
  int nlayers = source_net.layer_size();
  std::vector<std::string> afterLstmOutputs;

  for (int i =0; i<nlayers; ++i)
    {
      caffe::LayerParameter lparam = source_net.layer(i);
      if (lparam.type() == "MemoryData")
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
      else if (lparam.type() == "LSTM")
	{
	  lstmOutputs.push_back(lparam.top())
	  // simply skip this layer, it will be added via c++ api later
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

  bool TRTReadProtoFromTextFile(const char* filename, google::protobuf::Message* proto)
{   
  int fd = open(filename, O_RDONLY);
  if (fd == -1)
    return false;
  FileInputStream* input = new FileInputStream(fd);                                                     
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
  FileOutputStream* output = new FileOutputStream(fd);
  bool success = google::protobuf::TextFormat::Print(proto, output);
  delete output;                                                                                        
  close(fd);
  return success;
}

}
