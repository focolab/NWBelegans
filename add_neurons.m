s =genpath('../foco_lab/NWBelegans');
addpath(s);
t = genpath('data/atlases');
addpath(t);
u = genpath('../NP_eval');
addpath(u);

%%

%folder = '/Users/danielysprague/foco_lab/data/Manual_annotate/20230510-12-53-34';
%folder = '/Users/danielysprague/foco_lab/data/Manual_annotate/20230510-13-25-46';
%folder = '/Users/danielysprague/foco_lab/data/Manual_annotate/20230510-15-49-47';
%folder = '/Users/danielysprague/foco_lab/data/Manual_annotate/20230510-16-36-46';
%folder = '/Users/danielysprague/foco_lab/data/Manual_annotate/20221215-22-02-55';
%folder = '/Users/danielysprague/foco_lab/data/Manual_annotate/20230412-20-15-17';
%folder = '/Users/danielysprague/foco_lab/data/Manual_annotate/20230904-14-30-52';
%folder = '/Users/danielysprague/foco_lab/data/Manual_annotate/20230904-15-09-05';
%folder = '/Users/danielysprague/foco_lab/data/Manual_annotate/20230904-15-59-40';
%folder = '/Users/danielysprague/foco_lab/data/Manual_annotate/20230909-14-26-56';
%folder = '/Users/danielysprague/foco_lab/data/Manual_annotate/20230909-15-40-07';
%folder = '/Users/danielysprague/foco_lab/data/Manual_annotate/20230909-16-48-09';
%folder = '/Users/danielysprague/foco_lab/data/Manual_annotate/20240629-20-59-45';


datafile = load(strcat(folder, '/neuroPAL_image.mat'));

%histfile = load(strcat(folder, '/hist_med_image.mat'));

olddata = datafile.data;
info = datafile.info;
prefs = datafile.prefs;
version = datafile.version;
worm = datafile.worm;

%histRGBW = histfile.Hist_RGBW;

%olddata(:,:,:,1:4) = histRGBW;
data = olddata;

save(strcat(folder, '/neuroPAL_image.mat'), "data", "info", "prefs", "version", "worm")


%%

%folder = '/Users/danielysprague/foco_lab/data/Manual_annotate/20230510-12-53-34';
%folder = '/Users/danielysprague/foco_lab/data/Manual_annotate/20230510-13-25-46';
%folder = '/Users/danielysprague/foco_lab/data/Manual_annotate/20230510-15-49-47';
%folder = '/Users/danielysprague/foco_lab/data/Manual_annotate/20230510-16-36-46';
%folder = '/Users/danielysprague/foco_lab/data/Manual_annotate/20221215-22-02-55';
%folder = '/Users/danielysprague/foco_lab/data/Manual_annotate/20230412-20-15-17';
%folder = '/Users/danielysprague/foco_lab/data/Manual_annotate/20240629-20-59-45';
%folder = '/Users/danielysprague/foco_lab/data/Manual_annotate/20240626-12-35-05';
%folder = '/Users/danielysprague/foco_lab/data/Manual_annotate/20240626-13-55-40';
%folder = '/Users/danielysprague/foco_lab/data/Manual_annotate/20240629-14-14-08';
folder = '/Users/danielysprague/foco_lab/data/Manual_annotate/20240629-17-55-55';
%folder = '/Users/danielysprague/foco_lab/data/Manual_annotate/20240629-20-22-10';
%folder = '/Users/danielysprague/foco_lab/data/Manual_annotate/20240629-14-48-32';

ID_file = load(strcat(folder, '/neuroPAL_image_ID.mat'));

imfile = strcat(folder, '/neuroPAL_image.tif');

NP_image = DataHandling.NeuroPALImage;

[data, info, prefs, worm, mp, neurs, np_file, id_file] = NP_image.open(imfile);

dims = size(data);

Xmax= dims(2);
Ymax = dims(1);
Zmax = dims(3);

mp_params = ID_file.mp_params;
neurons = ID_file.neurons;
version = ID_file.version;
 
neurons.bodypart = 'head';

neuron_CSV = readtable(strcat(folder, '/blobs.csv'));

data_RGBW = double(data(:,:,:, [1,2,3,4]));
data_zscored = Methods.Preprocess.zscore_frame(double(data_RGBW));

for k =1:size(neuron_CSV,1)
    
    n = Neurons.Neuron;

    X = neuron_CSV{k, 3};
    Y = neuron_CSV{k, 4};
    Z = neuron_CSV{k, 5};
    
    n.position = [Y, X, Z];
    med_sample = data_zscored(round(Y)-1:round(Y)+1, round(X)-1:round(X)+1, round(Z), :);
    n.color_readout = squeeze(median(reshape(med_sample, [numel(med_sample)/size(med_sample,4), size(med_sample,4)])));
    
    neurons.neurons(k) = n;
    
end


save(strcat(folder, '/neuroPAL_image_ID.mat'), 'mp_params', 'neurons', 'version');

