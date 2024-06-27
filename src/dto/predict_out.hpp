/**
 * DeepDetect
 * Copyright (c) 2021 Jolibrain SASU
 * Author: Louis Jean <louis.jean@jolibrain.com>
 *
 * This file is part of deepdetect.
 *
 * deepdetect is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * deepdetect is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with deepdetect.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef DTO_PREDICT_OUT_HPP
#define DTO_PREDICT_OUT_HPP

#include <opencv2/opencv.hpp>
#ifdef USE_CUDA_CV
#include <opencv2/cudaimgproc.hpp>
#endif

#include "oatpp/core/Types.hpp"
#include "oatpp/core/macro/codegen.hpp"

#include "dto/common.hpp"
#include "dto/ddtypes.hpp"
#include "dto/resource.hpp"

namespace dd
{
  /** Data passed from an mllib to the next step in the chain */
  class ChainInputData
  {
  public:
    std::vector<cv::Mat> _imgs;
    std::vector<std::pair<int, int>> _img_sizes;
#ifdef USE_CUDA_CV
    std::vector<cv::cuda::GpuMat> _cuda_imgs;
#endif
  };

  namespace DTO
  {
#include OATPP_CODEGEN_BEGIN(DTO) ///< Begin DTO codegen section

    class PredictHead : public oatpp::DTO
    {
      DTO_INIT(PredictHead, DTO)

      DTO_FIELD(String, method) = "/predict";
      DTO_FIELD(String, service);

      DTO_FIELD(Int32, time);
    };

    class PredictClass : public oatpp::DTO
    {
      DTO_INIT(PredictClass, DTO)

      DTO_FIELD_INFO(last)
      {
        info->description
            = "If true, this is the last predicted class for this URI";
      }
      DTO_FIELD(Boolean, last);

      DTO_FIELD_INFO(bbox)
      {
        info->description = "Predicted bbox";
      }
      DTO_FIELD(Object<BBox>, bbox);

      DTO_FIELD_INFO(prob)
      {
        info->description = "Confidence / score for this class";
      }
      DTO_FIELD(Float32, prob);

      DTO_FIELD_INFO(cat)
      {
        info->description = "Class label";
      }
      DTO_FIELD(String, cat);

      DTO_FIELD_INFO(val)
      {
        info->description = "Regression value";
      }
      DTO_FIELD(Float32, val);

      DTO_FIELD_INFO(vals)
      {
        info->description = "Values for RoIs";
      }
      DTO_FIELD(DTOVector<double>, vals);

      DTO_FIELD_INFO(out)
      {
        info->description = "Output values for series";
      }
      DTO_FIELD(DTOVector<double>, out);

      DTO_FIELD_INFO(loss)
      {
        info->description = "Loss value for autoencoders";
      }
      DTO_FIELD(Float32, loss);

      DTO_FIELD_INFO(mask)
      {
        // XXX(louis): I don't know what this is
        info->description = "mask";
      }
      DTO_FIELD(DTOApiData, mask);

      DTO_FIELD_INFO(nns)
      {
        info->description = "[simsearch] Nearest neighbors";
      }
      DTO_FIELD(Vector<Any>, nns);

      /// XXX: May be removed when we get rid of APIData
      /// id to track class throught chains
      DTO_FIELD(String, class_id);
    };

    class Prediction : public oatpp::DTO
    {
      DTO_INIT(Prediction, DTO)

      DTO_FIELD(Boolean, last);

      DTO_FIELD_INFO(uri)
      {
        info->description
            = "URI to match this prediction with the corresponding input data";
      }
      DTO_FIELD(String, uri);

      DTO_FIELD_INFO(index_uri)
      {
        info->description = "[Simsearch]";
      }
      DTO_FIELD(String, index_uri);

      DTO_FIELD_INFO(classes)
      {
        info->description = "[Supervised] Array of predicted classes with "
                            "associated metadata (confidence, bbox, etc)";
      }
      DTO_FIELD(Vector<Object<PredictClass>>, classes)
          = Vector<Object<PredictClass>>::createShared();

      DTO_FIELD_INFO(series)
      {
        info->description = "[Supervised] series";
      }
      DTO_FIELD(Vector<Object<PredictClass>>, series);

      DTO_FIELD_INFO(vector)
      {
        info->description = "[Supervised] regression results";
      }
      DTO_FIELD(Vector<Object<PredictClass>>, vector);

      DTO_FIELD_INFO(losses)
      {
        info->description = "[Supervised] autoencoder";
      }
      DTO_FIELD(Vector<Object<PredictClass>>, losses);

      DTO_FIELD_INFO(rois)
      {
        info->description = "[Supervised] rois (?)";
      }
      DTO_FIELD(Vector<Object<PredictClass>>, rois);

      DTO_FIELD_INFO(vals)
      {
        info->description
            = "[Unsupervised] Array containing model output "
              "values. Can be in different formats: double, "
              "binarized double, booleans, binarized string, base64 image";
      }
      DTO_FIELD(Any, vals);

      DTO_FIELD_INFO(images)
      {
        info->description
            = "[Unsupervised] Array of images returned by the model";
      }
      DTO_FIELD(Vector<DTOImage>, images) = Vector<DTOImage>::createShared();

      DTO_FIELD_INFO(imgsize)
      {
        info->description = "[Unsupervised] Image size";
      }
      DTO_FIELD(Object<Dimensions>, imgsize);

      DTO_FIELD_INFO(confidences)
      {
        info->description
            = "[Unsupervised] Confidences per value of the vals vector";
      }
      DTO_FIELD(UnorderedFields<DTOVector<double>>, confidences);

      DTO_FIELD_INFO(indexed)
      {
        info->description
            = "[Simsearch] Whether indexed or not"; // XXX(louis) need more
                                                    // precise documentation
      }
      DTO_FIELD(Boolean, indexed);

      DTO_FIELD_INFO(nns)
      {
        info->description = "[simsearch] Nearest neighbors";
      }
      DTO_FIELD(Vector<Any>, nns);

      DTO_FIELD_INFO(loss)
      {
        // XXX(louis): I'm not sure of the documentation here
        info->description
            = "[Autoencoder][Legacy] Loss value for the only prediction";
      }
      DTO_FIELD(Float32, loss);

    public:
      // XXX: Legacy & deprecated
      std::vector<cv::Mat> _images; /**<allow to pass images in the DTO */
    };

    class PredictBody : public oatpp::DTO
    {
      DTO_INIT(PredictBody, DTO)

      DTO_FIELD_INFO(predictions)
      {
        info->description
            = "Array containing the prediction for each input data";
      }
      DTO_FIELD(Vector<Object<Prediction>>, predictions)
          = Vector<Object<Prediction>>::createShared();

      DTO_FIELD_INFO(time)
      {
        info->description = "Total prediction time";
      }
      DTO_FIELD(Float64, time);

      DTO_FIELD_INFO(measure)
      {
        info->description = "Measures if predict is launched as model test";
      }
      DTO_FIELD(DTOApiData, measure);

      DTO_FIELD_INFO(resources)
      {
        info->description
            = "Array containing resources status for each resource input data";
      }
      DTO_FIELD(Vector<Object<ResourceResponseBody>>, resources);

    public:
      /// chain input data
      ChainInputData _chain_input;
    };

    class PredictResponse : public oatpp::DTO
    {
      DTO_INIT(PredictResponse, DTO)

      DTO_FIELD(String, dd_msg);
      DTO_FIELD(Object<Status>, status);
      DTO_FIELD(Object<PredictHead>, head);
      DTO_FIELD(Object<PredictBody>, body);
    };

#include OATPP_CODEGEN_END(DTO) ///< End DTO codegen section
  }
}

#endif // DTO_PREDICT_OUT_HPP
