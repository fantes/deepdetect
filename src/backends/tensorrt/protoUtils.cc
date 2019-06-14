
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


nvinfer1::ILayer* findLayerByName(const nvinfer1::INetworkDefinition* network, const std::string lname)
{
  for (int i =0; i< network->getNbLayers(); ++i)
    {
      nvinfer1::ILayer * l = network->getLayer(i);
      if (l->getName() == lname)
	return l;
    }
  return nullptr;
}

  
void addUnparsablesFromProto(nvinfer1::INetworkDefinition* network, const std::string source_proto,
			     const std::string binary_proto,
			    const nvcaffeparser1::IBlobNameToTensor* b2t,
			    spdlog::logger* logger)
{
  caffe::NetParameter source_net;
  caffe::NetParameter binary_net;
  std::map<std::string, nvinfer1::ITensor*> t2t;
  if (!TRTReadProtoFromTextFile(source_proto.c_str(),&source_net))
    logger->error("TRT could read source protofile {}", source_proto);
  throw MLLibInternalException("TRT could read source protofile");
  if (!TRTReadProtoFromBinaryFile(source_proto.c_str(),&binary_net))
    logger->error("TRT could read net weights {}", binary_proto);
  throw MLLibInternalException("TRT could read net weights");
  int lstm_index = 0;
  int seq_size = -1;
  nvinfer1::ITensor * inputTensor = nullptr;
  for (int i =0; i<source_net.layers_size(); ++i)
    {
      caffe::LayerParameter lparam = source_net.layer(i);
      if (lparam.type() == "ContinuationIndicator")
	{
	  seq_size = lparam.continuation_indicator_param().time_step();
	}
      else if (lparam.type() == "LSTM")
	{
	  if (lstm_index == 0)
	    inputTensor = b2t->find(lparam.bottom(0).c_str());
	  int num_out = lparam.recurrent_param().num_output();
	  auto rnn = network->addRNNv2(*inputTensor, 1, num_out, seq_size, nvinfer1::RNNOperation::kLSTM);
	  const caffe::LayerParameter & binlayer = binary_net.layer(i);
	  const caffe::BlobProto& weight_blob = binlayer.blobs(0);
	  std::cout << "number of blobs in lstm layer: " << binlayer.blobs_size() << std::endl;
	  
	  int datasize = weight_blob.data_size() / num_out / 4;
	  
	  const float * weights = weight_blob.data().data();
	   
	  std::vector<float> zeros(num_out, 0.0);
	  nvinfer1::Weights iww{nvinfer1::DataType::kFLOAT,weights,num_out * datasize};
	  nvinfer1::Weights iwr{nvinfer1::DataType::kFLOAT,weights,num_out * datasize};
	  nvinfer1::Weights ibw{nvinfer1::DataType::kFLOAT,weights,num_out};
	  nvinfer1::Weights ibr{nvinfer1::DataType::kFLOAT,zeros.data(),num_out};
	  nvinfer1::Weights fww{nvinfer1::DataType::kFLOAT,weights,num_out * datasize};
	  nvinfer1::Weights fwr{nvinfer1::DataType::kFLOAT,weights,num_out * datasize};
	  nvinfer1::Weights fbw{nvinfer1::DataType::kFLOAT,weights,num_out};
	  nvinfer1::Weights fbr{nvinfer1::DataType::kFLOAT,zeros.data(),num_out};
	  nvinfer1::Weights oww{nvinfer1::DataType::kFLOAT,weights,num_out * datasize};
	  nvinfer1::Weights owr{nvinfer1::DataType::kFLOAT,weights,num_out * datasize};
	  nvinfer1::Weights obw{nvinfer1::DataType::kFLOAT,weights,num_out};
	  nvinfer1::Weights obr{nvinfer1::DataType::kFLOAT,zeros.data(),num_out};
	  nvinfer1::Weights cww{nvinfer1::DataType::kFLOAT,weights,num_out * datasize};
	  nvinfer1::Weights cwr{nvinfer1::DataType::kFLOAT,weights,num_out * datasize};
	  nvinfer1::Weights cbw{nvinfer1::DataType::kFLOAT,weights,num_out};
	  nvinfer1::Weights cbr{nvinfer1::DataType::kFLOAT,zeros.data(),num_out};

	  rnn->setWeightsForGate(0, nvinfer1::RNNGateType::kINPUT, true, iww);
	  rnn->setBiasForGate(0, nvinfer1::RNNGateType::kINPUT, true, ibw);
	  rnn->setWeightsForGate(0, nvinfer1::RNNGateType::kINPUT, false, iwr);
	  rnn->setBiasForGate(0, nvinfer1::RNNGateType::kINPUT, false, ibr);

	  rnn->setWeightsForGate(0, nvinfer1::RNNGateType::kFORGET, true, fww);
	  rnn->setBiasForGate(0, nvinfer1::RNNGateType::kFORGET, true, fbw);
	  rnn->setWeightsForGate(0, nvinfer1::RNNGateType::kFORGET, false, fwr);
	  rnn->setBiasForGate(0, nvinfer1::RNNGateType::kFORGET, false, fbr);


	  rnn->setWeightsForGate(0, nvinfer1::RNNGateType::kOUTPUT, true, oww);
	  rnn->setBiasForGate(0, nvinfer1::RNNGateType::kOUTPUT, true, obw);
	  rnn->setWeightsForGate(0, nvinfer1::RNNGateType::kOUTPUT, false, owr);
	  rnn->setBiasForGate(0, nvinfer1::RNNGateType::kOUTPUT, false, obr);

	  rnn->setWeightsForGate(0, nvinfer1::RNNGateType::kCELL, true, cww);
	  rnn->setBiasForGate(0, nvinfer1::RNNGateType::kCELL, true, cbw);
	  rnn->setWeightsForGate(0, nvinfer1::RNNGateType::kCELL, false, cwr);
	  rnn->setBiasForGate(0, nvinfer1::RNNGateType::kCELL, false, cbr);

	  std::string out = lparam.top(0);
	  inputTensor = rnn->getOutput(0);
	  t2t.insert(std::pair<std::string,nvinfer1::ITensor*>(out, inputTensor));
	  lstm_index++;
	}
      else
	{
	  for (int b = 0; b< lparam.bottom_size(); ++b)
	    {
	      std::map<std::string, nvinfer1::ITensor*>::iterator it = t2t.find(lparam.bottom(b));
	      if (it != t2t.end())
		{
		  // first find layer in already translated ones
		  nvinfer1::ILayer * atl = findLayerByName(network, lparam.name());
		  if (atl == nullptr)
		    {
		      logger->error("could not find layer {} for replacing input {} with output of manulaly created layer", lparam.name(), lparam.bottom(b));
		      throw MLLibInternalException("fatal error while creating network from caffe parser + manually adding layers");
		    }
		  // then update its input with ouput from manually inserter layer
		  atl->setInput(b,*(it->second));
		}
	    }
	}
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

std::vector<int> inputInList(caffe::LayerParameter&lparam, std::vector<std::string>list)
{
  std::vector<int> inList;
  for (int i =0; i< lparam.bottom_size(); ++i)
    if (std::find(list.begin(), list.end(), lparam.bottom(i)) != list.end())
      inList.push_back(i);
  return inList;
}

int fixProto(const std::string dest, const std::string source)
{
  caffe::NetParameter source_net;
  caffe::NetParameter dest_net;
  if (!TRTReadProtoFromTextFile(source.c_str(),&source_net))
    return 1;

  dest_net.set_name(source_net.name());
  std::vector<std::string> lstmOutputs;
  std::string rootInputName;
  
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
	  rootInputName = lparam.top(0);
	  dest_net.add_input(rootInputName);
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
	  lstmOutputs.push_back(lparam.top(0));
	}
      else
	{
	  caffe::LayerParameter* dlparam = dest_net.add_layer();
	  *dlparam = lparam;
	  std::vector<int> orphanedInputs = inputInList(lparam, lstmOutputs);
	  if (orphanedInputs.size() !=0)
	    {
	      for (int s:orphanedInputs)
		{
		  dlparam->set_bottom(s,rootInputName);
		}
	    }
	
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
  ZeroCopyInputStream* raw_input = new FileInputStream(fd);
  CodedInputStream* coded_input = new CodedInputStream(raw_input);
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
