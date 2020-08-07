#ifndef NBEATS_H
#define NBEATS_H

#include "torch/torch.h"
#include "../torchinputconns.h"

#include "native_net.h"

namespace dd
{
  class NBeats : public NativeModule
  {

	enum BlockType
	  {seasonality, trend, generic};


	class Block : public torch::nn::Module
	{
	public:
	  Block(int units, int thetas_dim,
			int backcast_length, int forecast_length,
			bool share_thetas) :
		_units(units), _thetas_dim(thetas_dim), _backcast_length(backcast_length),
		_forecast_length(forecast_length), _share_thetas(share_thetas)
	  {
		init_block();
	  }

	  torch::Tensor forward(torch::Tensor x);

	protected:

	  void init_block();

	  int _units;
	  int _thetas_dim;
	  int _backcast_length;
	  int _forecast_length;
	  bool _share_thetas;
	  std::vector<float> _backcast_linspace;
	  std::vector<float> _forecast_linspace;
	  torch::nn::Linear _fc1{nullptr};
	  torch::nn::Linear _fc2{nullptr};
	  torch::nn::Linear _fc3{nullptr};
	  torch::nn::Linear _fc4{nullptr};
	  torch::nn::Linear _theta_b_fc{nullptr};
	  torch::nn::Linear _theta_f_fc{nullptr};
	};

	class SeasonalityBlock : public Block
	{
	public:
	  SeasonalityBlock(int units, int thetas_dim,
					   int backcast_length, int forecast_length) :
		Block(units, thetas_dim, backcast_length, forecast_length, true) {}
	  std::tuple<torch::Tensor,torch::Tensor> forward(torch:: Tensor x);
	protected:
	  torch::Tensor seasonality_model(torch::Tensor x, const std::vector<float>& times);
	};

	class TrendBlock : public Block
	{
	public:
	  TrendBlock(int units, int thetas_dim,
				 int backcast_length, int forecast_length):
		Block(units, thetas_dim, backcast_length, forecast_length, true) {}
	  std::tuple<torch::Tensor,torch::Tensor> forward(torch:: Tensor x);
	protected:
	  torch::Tensor trend_model(torch::Tensor x, const std::vector<float>& times);
	};

	class GenericBlock: public Block
	{
	  GenericBlock(int units, int thetas_dim,
				   int backcast_length, int forecast_length):
		Block(units, thetas_dim, backcast_length, forecast_length, false)
	  {
		_fc1 = register_module("fc1", torch::nn::Linear(_backcast_length, _units));
		_fc2 = register_module("fc2", torch::nn::Linear(_units, _units));

	  }
	  std::tuple<torch::Tensor,torch::Tensor> forward(torch:: Tensor x);
	protected:
	  torch::nn::Linear _backcast_fc{nullptr};
	  torch::nn::Linear _forecast_fc{nullptr};
	};

	typedef std::vector<Block> Stack;



  public:

	NBeats(const APIData& adlib, const CSVTSTorchInputFileConn &inputc,
		   std::vector<BlockType> stackTypes = {trend, seasonality, generic},
		   int nb_blocks_per_stack = 3,
		   int forecast_length = 10, int backcast_length = 50,
		   std::vector<int> thetas_dims = {2,8,3},
		   bool share_weights_in_stack = false, int hidden_layer_units = 1024):
	  _forecast_length(forecast_length),
	  _backcast_length(backcast_length),
	  _hidden_layer_units(hidden_layer_units),
	  _nb_blocks_per_stack(nb_blocks_per_stack),
	  _share_weights_in_stack(share_weights_in_stack),
	  _stack_types(stackTypes),
	  _thetas_dims(thetas_dims)
	{
	  update_params(adlib, inputc);
	  create_nbeats();
	}
	NBeats():
	  _forecast_length(10), _backcast_length(50),
	  _hidden_layer_units(1024), _nb_blocks_per_stack(3),
	  _share_weights_in_stack(false),
	  _stack_types({trend, seasonality, generic}),
	  _thetas_dims({2,8,3})
	{
	  create_nbeats();
	}

	NBeats(std::vector<BlockType> stackTypes, int nb_blocks_per_stack,
		   int forecast_length, int backcast_length, std::vector<int> thetas_dims,
		   bool share_weights_in_stack, int hidden_layer_units):
	  _forecast_length(forecast_length), _backcast_length(backcast_length),
	  _hidden_layer_units(hidden_layer_units), _nb_blocks_per_stack(nb_blocks_per_stack),
	  _share_weights_in_stack(share_weights_in_stack),
	  _stack_types(stackTypes),
	  _thetas_dims(thetas_dims)
	{
	  create_nbeats();
	}


	virtual torch::Tensor forward(torch::Tensor x);

	virtual ~NBeats() {}

  protected:

	int _forecast_length;
	int _backcast_length;
	int _hidden_layer_units;
	int _nb_blocks_per_stack;
	bool _share_weights_in_stack;
	std::vector<BlockType> _stack_types;
	std::vector<Stack> _stacks;
	std::vector<int> _thetas_dims;

	void create_nbeats();
	void update_params(const APIData&adlib, const CSVTSTorchInputFileConn&inputc);


  };
}
#endif
