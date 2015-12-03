#!/usr/bin/perl -w
use File::Path qw(make_path remove_tree);
use List::Util qw(shuffle);
use File::Copy;

# (1) quit unless we have the correct number of command-line args
$num_args = $#ARGV + 1;
if ($num_args != 8) {
    print "\nUsage: createLastTwoLayerTuningCNNFiles.pl example_dir data_dir train_root model_name num_samples learning_rate iterations num_classes\n";
    exit;
}
 
# (2) we got 7 command line args, so assume they are the
$example_dir=$ARGV[0];
$data_dir=$ARGV[1];
$train_root=$ARGV[2];
$model_name=$ARGV[3];
$num_samples=$ARGV[4];
$learning_rate=$ARGV[5];
$iterations=$ARGV[6];
$num_classes=$ARGV[7];
 
# (3) print command line args to validate for user
print "Program options:\n";
print "\tExample_Dir: $example_dir\n\tData_Dir: $data_dir\n\tTrain_Root: $train_root\n\tModel_Name: $model_name\n\tNum_Samples: $num_samples\n\tLearning_Rate: $learning_rate\n\tIterations: $iterations\n";

# (4) create model database lvdb creation and mean image scripts for model 
$createSHText = "#!/usr/bin/env sh
# Create the $model_name lmdb inputs
# N.B. set the path to the $model_name train + val data dirs

EXAMPLE=/Users/Zach/Desktop/GraduateWork/ComputerVision/Project/CNNs/$example_dir/LastTwoLayerTuning/$num_samples
DATA=/Users/Zach/Desktop/GraduateWork/ComputerVision/Project/datasets/$data_dir/$num_samples
TOOLS=/Users/Zach/Desktop/GraduateWork/ComputerVision/Project/caffe/tools

TRAIN_DATA_ROOT=/Users/Zach/Desktop/GraduateWork/ComputerVision/Project/datasets/$train_root/
VAL_DATA_ROOT=/Users/Zach/Desktop/GraduateWork/ComputerVision/Project/datasets/$train_root/

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=true
if ".'$RESIZE'."; then
  RESIZE_HEIGHT=256
  RESIZE_WIDTH=256
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d \"".'$TRAIN_DATA_ROOT'."\" ]; then
  echo \"Error: TRAIN_DATA_ROOT is not a path to a directory: ".'$TRAIN_DATA_ROOT'."\"
  echo \"Set the TRAIN_DATA_ROOT variable in create_$model_name.sh to the path\" \\
       \"where the $model_name training data is stored.\"
  exit 1
fi

if [ ! -d \"".'$VAL_DATA_ROOT'."\" ]; then
  echo \"Error: VAL_DATA_ROOT is not a path to a directory: ".'$VAL_DATA_ROOT'."\"
  echo \"Set the VAL_DATA_ROOT variable in create_$model_name.sh to the path\" \\
       \"where the $model_name validation data is stored.\"
  exit 1
fi

echo \"Creating train lmdb...\"

GLOG_logtostderr=1 ".'$TOOLS'."/convert_imageset \\
    --resize_height=".'$RESIZE_HEIGHT'." \\
    --resize_width=".'$RESIZE_WIDTH'." \\
    --shuffle \\
    ".'$TRAIN_DATA_ROOT'." \\
    ".'$DATA'."/train.txt \\
    ".'$EXAMPLE'."/"."$model_name"."_train_lmdb

echo \"Creating val lmdb...\"

GLOG_logtostderr=1 ".'$TOOLS'."/convert_imageset \\
    --resize_height=".'$RESIZE_HEIGHT'." \\
    --resize_width=".'$RESIZE_WIDTH'." \\
    --shuffle \\
    ".'$VAL_DATA_ROOT'." \\
    ".'$DATA'."/val.txt \\
    ".'$EXAMPLE'."/"."$model_name"."_val_lmdb

echo \"Done.\"";
open($fh, '>', "$model_name/create_"."$model_name"."_$num_samples.sh");
print $fh "$createSHText";
close $fh;
$createMeanSHText = "#!/usr/bin/env sh
# Compute the mean image from the $model_name lmdb

EXAMPLE=/Users/Zach/Desktop/GraduateWork/ComputerVision/Project/CNNs/$example_dir/LastTwoLayerTuning/$num_samples
DATA=/Users/Zach/Desktop/GraduateWork/ComputerVision/Project/datasets/$data_dir/$num_samples
TOOLS=/Users/Zach/Desktop/GraduateWork/ComputerVision/Project/caffe/tools

".'$TOOLS'."/compute_image_mean ".'$EXAMPLE'."/"."$model_name"."_train_lmdb \
  ".'$DATA'."/"."$model_name"."_mean.binaryproto

echo \"Done.\"";
open($fh, '>', "$model_name/make_"."$model_name"."_mean_$num_samples.sh");
print $fh "$createMeanSHText";
close $fh;

# (4) remaining set-up steps, remove old folders and files, and create new directories
if(-d "/Users/Zach/Desktop/GraduateWork/ComputerVision/Project/CNNs/$example_dir/LastTwoLayerTuning/$num_samples/"."$model_name"."_train_lmdb"){
	remove_tree("/Users/Zach/Desktop/GraduateWork/ComputerVision/Project/CNNs/$example_dir/LastTwoLayerTuning/$num_samples/"."$model_name"."_train_lmdb");
}
if(-d "/Users/Zach/Desktop/GraduateWork/ComputerVision/Project/CNNs/$example_dir/LastTwoLayerTuning/$num_samples/"."$model_name"."_val_lmdb"){
	remove_tree("/Users/Zach/Desktop/GraduateWork/ComputerVision/Project/CNNs/$example_dir/LastTwoLayerTuning/$num_samples/"."$model_name"."_val_lmdb");
}
if(-d "/Users/Zach/Desktop/GraduateWork/ComputerVision/Project/datasets/$data_dir/$num_samples"){
	remove_tree("/Users/Zach/Desktop/GraduateWork/ComputerVision/Project/datasets/$data_dir/$num_samples");
}
mkdir "/Users/Zach/Desktop/GraduateWork/ComputerVision/Project/CNNs/$example_dir" unless -d "/Users/Zach/Desktop/GraduateWork/ComputerVision/Project/CNNs/$example_dir";
mkdir "/Users/Zach/Desktop/GraduateWork/ComputerVision/Project/CNNs/$example_dir/LastTwoLayerTuning" unless -d "Users/Zach/Desktop/GraduateWork/ComputerVision/Project/CNNs/$example_dir/LastTwoLayerTuning";
mkdir "/Users/Zach/Desktop/GraduateWork/ComputerVision/Project/CNNs/$example_dir/LastTwoLayerTuning/$num_samples" unless -d "/Users/Zach/Desktop/GraduateWork/ComputerVision/Project/CNNs/$example_dir/LastTwoLayerTuning/$num_samples";
mkdir "/Users/Zach/Desktop/GraduateWork/ComputerVision/Project/datasets/$data_dir/$num_samples" unless -d "/Users/Zach/Desktop/GraduateWork/ComputerVision/Project/datasets/$data_dir/$num_samples";

# (5) create train/val .txt files and synset_words.txt file
opendir(D, "/Users/Zach/Desktop/GraduateWork/ComputerVision/Project/datasets/$train_root") || die "Can't open directory $d: $!\n";
my @directoryList = readdir(D);
closedir(D);
open(my $train_file, '>', "/Users/Zach/Desktop/GraduateWork/ComputerVision/Project/datasets/$data_dir/$num_samples/train.txt");
open(my $val_file, '>', "/Users/Zach/Desktop/GraduateWork/ComputerVision/Project/datasets/$data_dir/$num_samples/val.txt");
open(my $test_file, '>', "/Users/Zach/Desktop/GraduateWork/ComputerVision/Project/datasets/$data_dir/$num_samples/test.txt");
open(my $synset_words, '>', "/Users/Zach/Desktop/GraduateWork/ComputerVision/Project/CNNs/$example_dir/LastTwoLayerTuning/$num_samples/synset_words.txt");
$labelCount = 0;
foreach my $directory (@directoryList) {
	next if(($directory eq "..")||($directory eq ".")||($directory eq ".DS_Store"));
	opendir(imageDir, "/Users/Zach/Desktop/GraduateWork/ComputerVision/Project/datasets/$train_root/$directory") || die "Can't open directory $directory: $!\n";
	my @imageList = readdir(imageDir);
	closedir(imageDir);
	shift @imageList;
	shift @imageList;
	$imageCount = 0+@imageList;
	@imageListRandom = shuffle(@imageList);
	for($i=0; $i<$imageCount-1;$i++){
		if($i<$num_samples){
			print $train_file "$directory/$imageListRandom[$i] $labelCount\n";
			print $test_file "/Users/Zach/Desktop/GraduateWork/ComputerVision/Project/datasets/$train_root/$directory/$imageListRandom[$i] $labelCount\n";
			$i++;
			print $val_file "$directory/$imageListRandom[$i] $labelCount\n";
			print $test_file "/Users/Zach/Desktop/GraduateWork/ComputerVision/Project/datasets/$train_root/$directory/$imageListRandom[$i] $labelCount\n";
		}
		print $test_file "/Users/Zach/Desktop/GraduateWork/ComputerVision/Project/datasets/$train_root/$directory/$imageListRandom[$i] $labelCount\n";
	}
	print $synset_words "$directory\n";
	$labelCount++;
}
close $train_file;
close $val_file;
close $test_file;
close $synset_words;

# (6) create solver.prototxt
$test_interval = $iterations/10;
$solverText = "net: \""."$example_dir/LastTwoLayerTuning/$num_samples/"."train_val_"."$learning_rate"."_$iterations.prototxt\"
test_iter: 10
test_interval: "."$test_interval"."
base_lr: "."$learning_rate"."
lr_policy: \"step\"
gamma: 0.1
stepsize: "."$test_interval"."
display: 1
max_iter: "."$iterations"."
momentum: 0.9
weight_decay: 0.0005
snapshot: "."$iterations"."
snapshot_prefix: \"$example_dir/LastTwoLayerTuning/$num_samples/"."$model_name"."_$learning_rate"."_$iterations"."_train"."\"
solver_mode: CPU";
open($fh, '>', "/Users/Zach/Desktop/GraduateWork/ComputerVision/Project/CNNs/$example_dir/LastTwoLayerTuning/$num_samples/solver_"."$learning_rate"."_$iterations.prototxt");
print $fh "$solverText";
close $fh;

# (7) create train_val.prototxt
$train_valText = "name: \""."$model_name"."\"
layer {
  name: \"data\"
  type: \"Data\"
  top: \"data\"
  top: \"label\"
  include {
    phase: TRAIN
  }
  transform_param {
    mirror: true
    crop_size: 227
    mean_file: \"imagenet_mean.binaryproto\"
  }
  data_param {
    source: \"$example_dir/LastTwoLayerTuning/$num_samples/"."$model_name"."_train_lmdb\"
    batch_size: 50
    backend: LMDB
  }
}
layer {
  name: \"data\"
  type: \"Data\"
  top: \"data\"
  top: \"label\"
  include {
    phase: TEST
  }
  transform_param {
    mirror: false
    crop_size: 227
    mean_file: \"imagenet_mean.binaryproto\"
  }
  data_param {
    source: \"$example_dir/LastTwoLayerTuning/$num_samples/"."$model_name"."_val_lmdb\"
    batch_size: 50
    backend: LMDB
  }
}
layer {
  name: \"conv1\"
  type: \"Convolution\"
  bottom: \"data\"
  top: \"conv1\"
  param {
    lr_mult: 0
    decay_mult: 1
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  convolution_param {
    num_output: 96
    kernel_size: 11
    stride: 4
    weight_filler {
      type: \"gaussian\"
      std: 0.01
    }
    bias_filler {
      type: \"constant\"
      value: 0
    }
  }
}
layer {
  name: \"relu1\"
  type: \"ReLU\"
  bottom: \"conv1\"
  top: \"conv1\"
}
layer {
  name: \"pool1\"
  type: \"Pooling\"
  bottom: \"conv1\"
  top: \"pool1\"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: \"norm1\"
  type: \"LRN\"
  bottom: \"pool1\"
  top: \"norm1\"
  lrn_param {
    local_size: 5
    alpha: 0.0001
    beta: 0.75
  }
}
layer {
  name: \"conv2\"
  type: \"Convolution\"
  bottom: \"norm1\"
  top: \"conv2\"
  param {
    lr_mult: 0
    decay_mult: 1
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  convolution_param {
    num_output: 256
    pad: 2
    kernel_size: 5
    group: 2
    weight_filler {
      type: \"gaussian\"
      std: 0.01
    }
    bias_filler {
      type: \"constant\"
      value: 1
    }
  }
}
layer {
  name: \"relu2\"
  type: \"ReLU\"
  bottom: \"conv2\"
  top: \"conv2\"
}
layer {
  name: \"pool2\"
  type: \"Pooling\"
  bottom: \"conv2\"
  top: \"pool2\"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: \"norm2\"
  type: \"LRN\"
  bottom: \"pool2\"
  top: \"norm2\"
  lrn_param {
    local_size: 5
    alpha: 0.0001
    beta: 0.75
  }
}
layer {
  name: \"conv3\"
  type: \"Convolution\"
  bottom: \"norm2\"
  top: \"conv3\"
  param {
    lr_mult: 0
    decay_mult: 1
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  convolution_param {
    num_output: 384
    pad: 1
    kernel_size: 3
    weight_filler {
      type: \"gaussian\"
      std: 0.01
    }
    bias_filler {
      type: \"constant\"
      value: 0
    }
  }
}
layer {
  name: \"relu3\"
  type: \"ReLU\"
  bottom: \"conv3\"
  top: \"conv3\"
}
layer {
  name: \"conv4\"
  type: \"Convolution\"
  bottom: \"conv3\"
  top: \"conv4\"
  param {
    lr_mult: 0
    decay_mult: 1
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  convolution_param {
    num_output: 384
    pad: 1
    kernel_size: 3
    group: 2
    weight_filler {
      type: \"gaussian\"
      std: 0.01
    }
    bias_filler {
      type: \"constant\"
      value: 1
    }
  }
}
layer {
  name: \"relu4\"
  type: \"ReLU\"
  bottom: \"conv4\"
  top: \"conv4\"
}
layer {
  name: \"conv5\"
  type: \"Convolution\"
  bottom: \"conv4\"
  top: \"conv5\"
  param {
    lr_mult: 0
    decay_mult: 1
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  convolution_param {
    num_output: 256
    pad: 1
    kernel_size: 3
    group: 2
    weight_filler {
      type: \"gaussian\"
      std: 0.01
    }
    bias_filler {
      type: \"constant\"
      value: 1
    }
  }
}
layer {
  name: \"relu5\"
  type: \"ReLU\"
  bottom: \"conv5\"
  top: \"conv5\"
}
layer {
  name: \"pool5\"
  type: \"Pooling\"
  bottom: \"conv5\"
  top: \"pool5\"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: \"fc6\"
  type: \"InnerProduct\"
  bottom: \"pool5\"
  top: \"fc6\"
  param {
    lr_mult: 0
    decay_mult: 1
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  inner_product_param {
    num_output: 4096
    weight_filler {
      type: \"gaussian\"
      std: 0.005
    }
    bias_filler {
      type: \"constant\"
      value: 1
    }
  }
}
layer {
  name: \"relu6\"
  type: \"ReLU\"
  bottom: \"fc6\"
  top: \"fc6\"
}
layer {
  name: \"drop6\"
  type: \"Dropout\"
  bottom: \"fc6\"
  top: \"fc6\"
  dropout_param {
    dropout_ratio: 0.5
  }
}
layer {
  name: \"fc7\"
  type: \"InnerProduct\"
  bottom: \"fc6\"
  top: \"fc7\"
  param {
    lr_mult: 10
    decay_mult: 1
  }
  param {
    lr_mult: 20
    decay_mult: 0
  }
  inner_product_param {
    num_output: 4096
    weight_filler {
      type: \"gaussian\"
      std: 0.005
    }
    bias_filler {
      type: \"constant\"
      value: 1
    }
  }
}
layer {
  name: \"relu7\"
  type: \"ReLU\"
  bottom: \"fc7\"
  top: \"fc7\"
}
layer {
  name: \"drop7\"
  type: \"Dropout\"
  bottom: \"fc7\"
  top: \"fc7\"
  dropout_param {
    dropout_ratio: 0.5
  }
}
layer {
  name: \"fc8-"."$model_name"."\"
  type: \"InnerProduct\"
  bottom: \"fc7\"
  top: \"fc8-"."$model_name"."\"
  param {
    lr_mult: 10
    decay_mult: 1
  }
  param {
    lr_mult: 20
    decay_mult: 0
  }
  inner_product_param {
    num_output: $labelCount
    weight_filler {
      type: \"gaussian\"
      std: 0.01
    }
    bias_filler {
      type: \"constant\"
      value: 0
    }
  }
}
layer {
  name: \"loss\"
  type: \"SoftmaxWithLoss\"
  bottom: \"fc8-"."$model_name"."\"
  bottom: \"label\"
  top: \"loss\"
}

layer {
  name: \"accuracy\"
  type: \"Accuracy\"
  bottom: \"fc8-"."$model_name"."\"
  bottom: \"label\"
  top: \"accuracy\"
  include {
    phase: TEST
  }
}
";
open($fh, '>', "/Users/Zach/Desktop/GraduateWork/ComputerVision/Project/CNNs/$example_dir/LastTwoLayerTuning/$num_samples/train_val_"."$learning_rate"."_$iterations.prototxt");
print $fh "$train_valText";
close $fh;

# (8) create deploy.prototxt
$deployText = "name: \"$model_name\"
input: \"data\"
input_dim: 10
input_dim: 3
input_dim: 227
input_dim: 227
layer {
  name: \"conv1\"
  type: \"Convolution\"
  bottom: \"data\"
  top: \"conv1\"
  convolution_param {
    num_output: 96
    kernel_size: 11
    stride: 4
  }
}
layer {
  name: \"relu1\"
  type: \"ReLU\"
  bottom: \"conv1\"
  top: \"conv1\"
}
layer {
  name: \"pool1\"
  type: \"Pooling\"
  bottom: \"conv1\"
  top: \"pool1\"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: \"norm1\"
  type: \"LRN\"
  bottom: \"pool1\"
  top: \"norm1\"
  lrn_param {
    local_size: 5
    alpha: 0.0001
    beta: 0.75
  }
}
layer {
  name: \"conv2\"
  type: \"Convolution\"
  bottom: \"norm1\"
  top: \"conv2\"
  convolution_param {
    num_output: 256
    pad: 2
    kernel_size: 5
    group: 2
  }
}
layer {
  name: \"relu2\"
  type: \"ReLU\"
  bottom: \"conv2\"
  top: \"conv2\"
}
layer {
  name: \"pool2\"
  type: \"Pooling\"
  bottom: \"conv2\"
  top: \"pool2\"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: \"norm2\"
  type: \"LRN\"
  bottom: \"pool2\"
  top: \"norm2\"
  lrn_param {
    local_size: 5
    alpha: 0.0001
    beta: 0.75
  }
}
layer {
  name: \"conv3\"
  type: \"Convolution\"
  bottom: \"norm2\"
  top: \"conv3\"
  convolution_param {
    num_output: 384
    pad: 1
    kernel_size: 3
  }
}
layer {
  name: \"relu3\"
  type: \"ReLU\"
  bottom: \"conv3\"
  top: \"conv3\"
}
layer {
  name: \"conv4\"
  type: \"Convolution\"
  bottom: \"conv3\"
  top: \"conv4\"
  convolution_param {
    num_output: 384
    pad: 1
    kernel_size: 3
    group: 2
  }
}
layer {
  name: \"relu4\"
  type: \"ReLU\"
  bottom: \"conv4\"
  top: \"conv4\"
}
layer {
  name: \"conv5\"
  type: \"Convolution\"
  bottom: \"conv4\"
  top: \"conv5\"
  convolution_param {
    num_output: 256
    pad: 1
    kernel_size: 3
    group: 2
  }
}
layer {
  name: \"relu5\"
  type: \"ReLU\"
  bottom: \"conv5\"
  top: \"conv5\"
}
layer {
  name: \"pool5\"
  type: \"Pooling\"
  bottom: \"conv5\"
  top: \"pool5\"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: \"fc6\"
  type: \"InnerProduct\"
  bottom: \"pool5\"
  top: \"fc6\"
  inner_product_param {
    num_output: 4096
  }
}
layer {
  name: \"relu6\"
  type: \"ReLU\"
  bottom: \"fc6\"
  top: \"fc6\"
}
layer {
  name: \"drop6\"
  type: \"Dropout\"
  bottom: \"fc6\"
  top: \"fc6\"
  dropout_param {
    dropout_ratio: 0.5
  }
}
layer {
  name: \"fc7\"
  type: \"InnerProduct\"
  bottom: \"fc6\"
  top: \"fc7\"
  inner_product_param {
    num_output: 4096
  }
}
layer {
  name: \"relu7\"
  type: \"ReLU\"
  bottom: \"fc7\"
  top: \"fc7\"
}
layer {
  name: \"drop7\"
  type: \"Dropout\"
  bottom: \"fc7\"
  top: \"fc7\"
  dropout_param {
    dropout_ratio: 0.5
  }
}
layer {
  name: \"fc8-"."$model_name"."\"
  type: \"InnerProduct\"
  bottom: \"fc7\"
  top: \"fc8-"."$model_name"."\"
  inner_product_param {
    num_output: $labelCount
  }
}
layer {
  name: \"prob\"
  type: \"Softmax\"
  bottom: \"fc8-"."$model_name"."\"
  top: \"prob\"
}";
open($fh, '>', "/Users/Zach/Desktop/GraduateWork/ComputerVision/Project/CNNs/$example_dir/LastTwoLayerTuning/$num_samples/deploy.prototxt");
print $fh "$deployText";
close $fh;

# (9) create lvdb for images
$command = "bash $model_name/create_"."$model_name"."_$num_samples".".sh";
system($command);

# (10) train caffe model
$command = "/Users/Zach/Desktop/GraduateWork/ComputerVision/Project/caffe/tools/caffe train -solver=/Users/Zach/Desktop/GraduateWork/ComputerVision/Project/CNNs/$example_dir/LastTwoLayerTuning/$num_samples/solver_"."$learning_rate"."_$iterations.prototxt -weights=/Users/Zach/Desktop/GraduateWork/ComputerVision/Project/CNNs/bvlc_reference_caffenet.caffemodel";
system($command);

# (11) test caffe model
$command = "/Users/Zach/Desktop/GraduateWork/ComputerVision/Project/caffe/examples/cpp_classification/classification /Users/Zach/Desktop/GraduateWork/ComputerVision/Project/CNNs/$example_dir/LastTwoLayerTuning/$num_samples/deploy.prototxt /Users/Zach/Desktop/GraduateWork/ComputerVision/Project/CNNs/$example_dir/LastTwoLayerTuning/$num_samples/"."$model_name"."_$learning_rate"."_$iterations"."_train"."_iter_1000.caffemodel /Users/Zach/Desktop/GraduateWork/ComputerVision/Project/CNNs/imagenet_mean.binaryproto /Users/Zach/Desktop/GraduateWork/ComputerVision/Project/CNNs/$example_dir/LastTwoLayerTuning/$num_samples/synset_words.txt /Users/Zach/Desktop/GraduateWork/ComputerVision/Project/datasets/"."$data_dir"."/$num_samples/test.txt $num_classes";
system($command);