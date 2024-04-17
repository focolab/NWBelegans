t = genpath('/Users/danielysprague/foco_lab/data/atlases');
addpath(t);

%folders = ["Yemini_NWB", "NWB_chaudhary", "NWB_foco", "NWB_Ray","NWB_NP_flavell"];

%folder = '/Users/danielysprague/foco_lab/data/Yemini_NWB';
%folder = '/Users/danielysprague/foco_lab/data/NWB_chaudhary';
%folder = '/Users/danielysprague/foco_lab/data/NWB_foco';
%folder = '/Users/danielysprague/foco_lab/data/NWB_Ray';
folder = '/Users/danielysprague/foco_lab/data/NP_nwb';
%folder = '/Users/danielysprague/foco_lab/data/NWB_NP_Flavell';

%dict_files = ["20221028-18-48-00" "20221106-21-00-09" "20221106-21-23-19" "20221106-21-47-31" "20221215-20-02-49" "20230322-18-57-04" "20230322-20-16-50" "20230322-21-41-10" "20230322-22-43-03" "20230412-20-15-17" "20230506-12-56-00" "20230506-14-24-57" "20230506-15-01-45" "20230506-15-33-51" "20230510-12-53-34" "20230510-13-25-46" "20230510-15-49-47" "20230904-14-30-52" "20230904-15-09-05" "20230904-15-59-40" "20230909-16-48-09"];
%dict_slices = ["300-578" "340-626" "337-610" "280-595" "500-780" "230-700" "280-700" "300-700" "385-710" "400-708" "642-1364" "708-1400" "580-1278" "636-1200" "688-1360" "620-1300" "626-1340" "632-1410" "510-1450" "636-1410" "605-1390"];
%crop_dict = dictionary(dict_files, dict_slices);

files = dir(folder);

for i = 1:size(files)

    file = files(i).name;
    if ~endsWith(file, '.nwb')
        continue
    end

    %if isfile(strcat('/Users/danielysprague/foco_lab/data/hist_matched_test2/',extractBefore(file,'.nwb'),'.mat'))
    %    continue
    %end

    disp(file);

    filepath = strcat(folder, '/',file);

    image_data = nwbRead(filepath);
    data = image_data.acquisition.get('NeuroPALImageRaw').data.load();

    identifier = image_data.identifier;

    data_order = 1:ndims(data);
    data_order(1) = 3;
    data_order(2) = 4;
    data_order(3) = 2;
    data_order(4) = 1;
    %data_order(1) = 2;
    %data_order(2) = 1;
    data = permute(data, data_order);
    
    %image_data.scale(1) = image_data.scale(2);
    %image_data.scale(2) = image_data.scale(2);
            
    % Setup the NP file data.
    info.file = filepath;
    neuroPAL_module = image_data.processing.get('NeuroPAL');
    %imagingVolume = image_data.general_optophysiology.get('NPImVol');
    imagingVolume = image_data.general_optophysiology.get('NeuroPALImVol');
    %imagingVolume = neuroPAL_module.nwbdatainterface.get('ImagingVolume');
    grid_spacing_data = imagingVolume.grid_spacing.load();
    info.scale = grid_spacing_data;
    info.DIC = nan;
            
    % Determine the color channels.
    %colors = image_data.colors;
    %colors = round(colors/max(colors(:)));
    npalraw = image_data.acquisition.get('NeuroPALImageRaw');
    rgbw = npalraw.RGBW_channels.load();
    info.RGBW = rgbw+1;
    info.GFP = nan;
    
    % Initialize the user preferences.
    prefs.RGBW = info.RGBW;
    prefs.rotate.horizontal = false;
    prefs.rotate.vertical = false;
    prefs.z_center = ceil(size(data,3) / 2);
    prefs.is_Z_LR = true;
    prefs.is_Z_flip = true;

    data = uint32(data);

    %if isKey(crop_dict, identifier);
    %    xcrops = split(crop_dict(identifier),"-");
    %    startx = str2num(xcrops(1));
    %    endx = str2num(xcrops(2));

    %    data = data(:, startx:endx,:,:);
    %    disp(size(data));
    %end

    %NP_file = '/Users/danielysprague/foco_lab/data/NP_paper/all/7_YAaLR.mat';

    %NP_image = DataHandling.NeuroPALImage;
    
    %[refdata, refinfo, refprefs, refworm, mp, refneurons, np_file, id_file] = NP_image.open(NP_file);

    %refdata = refdata(:,:,:,refprefs.RGBW);

    RGBW = prefs.RGBW;
        
    im = Neurons.Image([]);

    %autID = Methods.AutoId.instance();

    if size(data, 4) == 3

        data_RGB = data(:,:,:,RGBW(1:3));

        data_RGBW = zeros(size(data,1), size(data,2), size(data,3),4);
        data_RGBW(:,:,:,1:3) = data_RGB;

    else
        data_RGBW = data(:,:,:,RGBW);
    end

    data_RGBW = data_RGBW(:,:,:,1:3);

    data_matched = MatchHist(data_RGBW);

    
    save(strcat('/Users/danielysprague/foco_lab/data/hist_matched_test3/',extractBefore(file,'.nwb'),'.mat'), 'data_matched')

    %save(strcat('/Users/danielysprague/foco_lab/data/raw_neuropal/',extractBefore(file,'.nwb'),'.mat'), 'data_RGBW')

end