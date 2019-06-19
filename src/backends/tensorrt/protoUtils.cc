
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

  nvinfer1::ITensor* findInputTensorByName(const nvinfer1::INetworkDefinition* network, const std::string name)
  {
    for (int i=0; i<network->getNbInputs(); ++i)
      {
        nvinfer1::ITensor* it = network->getInput(i);
        if (it->getName() == name)
          return it;
      }
    return nullptr;
  }

  void visualizeNet(const nvinfer1::INetworkDefinition* network)
  {
    std::cout << "inputs: " << std::endl;
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
          {
            std::cout << "     input " << j << " " << l->getInput(j)->getName() << std::endl;
          }
        for (int j=0; j< nout; ++j)
          {
            std::cout << "     output " << j << " " << l->getOutput(j)->getName() << std::endl;
          }
      }

  }


void addUnparsablesFromProto(nvinfer1::INetworkDefinition* network,
                             const std::string& source_proto,
                             const std::string& binary_proto,
                             std::vector<int>& unparsable,
                             std::map<int,std::vector<int>>& tofix,
                             std::vector<std::string>& removedOutputs,
                             std::string& rooInputName,
                             const nvcaffeparser1::IBlobNameToTensor* b2t,
                             spdlog::logger* logger)
{
  caffe::NetParameter source_net;
  caffe::NetParameter binary_net;
  if (!TRTReadProtoFromTextFile(source_proto.c_str(),&source_net))
    {
      logger->error("TRT could read source protofile {}", source_proto);
      throw MLLibInternalException("TRT could read source protofile");
    }
  if (!TRTReadProtoFromBinaryFile(binary_proto.c_str(),&binary_net))
    {
      logger->error("TRT could read net weights {}", binary_proto);
      throw MLLibInternalException("TRT could read net weights");
    }

  std::map<std::string, nvinfer1::ITensor*> t2t;
  int lstm_index = 0;
  int seq_size = -1;
  nvinfer1::ITensor * inputTensor = nullptr;
  for (int i : unparsable)
    {
      caffe::LayerParameter lparam = source_net.layer(i);
      if (lparam.type() == "ContinuationIndicator")
        {
          seq_size = lparam.continuation_indicator_param().time_step();
        }
      else if (lparam.type() == "LSTM")
        {
          if (lstm_index == 0)
            {
              inputTensor = b2t->find(lparam.bottom(0).c_str());
              std::cout << "LSTM i0 found input tensor in caffe parsed "
                         << inputTensor->getName() << std::endl;
            }
          else
            {
              inputTensor = t2t[lparam.bottom(0)];
              std::cout << "LSTM " << lstm_index << " found input tensor in newly added ones "
                         << inputTensor->getName() << std::endl;
            }
          int num_out = lparam.recurrent_param().num_output();
          nvinfer1::IRNNv2Layer * rnn = network->addRNNv2(*inputTensor, 1, num_out, seq_size, nvinfer1::RNNOperation::kLSTM);
          std::stringstream n;
          n<< "lstm" << lstm_index;
          rnn->setName(n.str().c_str());
          const caffe::LayerParameter * binlayer = findLayerByName(binary_net,lparam.name());
          const caffe::BlobProto& weight_blob = binlayer->blobs(0);
          std::cout << "number of blobs in lstm layer: " << binlayer->blobs_size() << std::endl;

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
          nvinfer1::ITensor* loutput = rnn->getOutput(0);
          std::string out_real = out + "_real";
          loutput->setName(out_real.c_str());
          std::cout << "adding ouput tensor " << loutput->getName() << " to t2t" << std::endl;
          t2t.insert(std::pair<std::string,nvinfer1::ITensor*>(out, loutput));
          lstm_index++;
        }
      else
        {
          logger->error("unknow unparsable layer {} of type {}", lparam.name(), lparam.type());
          throw MLLibInternalException("fatal while trying to manually add layer");
        }
    }
  for (auto const& tf: tofix)
    {
      caffe::LayerParameter lparam = source_net.layer(tf.first);
      std::cout << "fixing layer " << tf.first << " " << lparam.name() << std::endl;
      for (int j: tf.second)
        {
          std::string bname = lparam.bottom(j);
          nvinfer1::ILayer* il = findLayerByName(network, lparam.name());
          il->setInput(j,*(t2t[bname]));
          std::cout << "fixing input " << bname << " of layer " << lparam.name() << " with " << t2t[bname]->getName() << std::endl;
        }
    }
  for (std::string ro : removedOutputs)
    {
      nvinfer1::ITensor* it = findInputTensorByName(network, ro);
      if (it != nullptr)
        {
          std::cout << "removing fake input " << ro << std::endl;
          network->removeTensor(*it);
        }
    }

  visualizeNet(network);
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

std::vector<int> inputInList(const caffe::LayerParameter&lparam, std::vector<std::string>list)
{
  std::vector<int> inList;
  for (int i =0; i< lparam.bottom_size(); ++i)
    if (std::find(list.begin(), list.end(), lparam.bottom(i)) != list.end())
      inList.push_back(i);
  return inList;
}

caffe::LayerParameter* findLayerByName(caffe::NetParameter& net, const std::string name)
{
  for (int i=0; i< net.layer_size(); ++i)
    {
      caffe::LayerParameter*lparam = net.mutable_layer(i);
      if (lparam->name() == name)
        return lparam;
    }
  return nullptr;
}


  int fixProto(const std::string dest, const std::string source,std::vector<int>&unparsable, std::map<int,std::vector<int>>&tofix, std::vector<std::string> & removedOutputs, std::string& rootInputName, const std::string binary_proto)
{
  caffe::NetParameter source_net;
  caffe::NetParameter dest_net;

  int timesteps;
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
          unparsable.push_back(i);
          timesteps = lparam.continuation_indicator_param().time_step();
          // simply skip this layer
        }
      else if (lparam.type() == "LSTM")
        {
          unparsable.push_back(i);
          removedOutputs.push_back(lparam.top(0));
          // add fake input
          std::cout << "adding fake input " << lparam.top(0) << std::endl;
          dest_net.add_input(lparam.top(0));
          caffe::BlobShape* is = dest_net.add_input_shape();
          const caffe::LayerParameter* binlayer = findLayerByName(binary_net, lparam.name());
          const caffe::BlobProto& weight_blob = binlayer->blobs(0);
          int num_out = lparam.recurrent_param().num_output();
          int datasize = weight_blob.data_size() / num_out / 4;
          is->add_dim(1); // bs will be overriden
          is->add_dim(1);
          is->add_dim(datasize);
          is->add_dim(1);
        }
      else 
        {
          caffe::LayerParameter* dlparam = dest_net.add_layer();
          *dlparam = lparam;
          std::vector<int> inputsInRemoved = inputInList(lparam, removedOutputs);
          if (inputsInRemoved.size() != 0)
            {
              std::cout << "adding layer " << i << " " << lparam.name() << " to tofix" << std::endl;
              tofix[i] = inputsInRemoved;
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
