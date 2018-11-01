%% --------------------------
% DRRN_B1U9
% -------------------------------
function test_filter_loopMany_decoder()
setenv('LC_ALL','C')
addpath /home/jiaby/caffe/matlab; % change to your caffe path
setenv('GLOG_minloglevel','2')
addpath('../');
addpath('../evaluation_func/');
addpath('../evaluation_func/matlabPyrTools-master/');

%% parameters
gpu_id = 1;




thresh_hei = 900; % threshold patch size for inference, since too big image may cost too much memory  160 262
thresh_wid = 900;
rf = 32;

gain = 0;

fid1 = fopen('/media/tjc/RSE/test/rse3/location_txt/merge_s10.txt','rt');   %%1111111111111111111111111111111111  mush be attention!!!

mean_SeFilter = [];
mean_HM_psnr = [];
mean_h1_psnr = [];
% caffe.set_mode_cpu(); % for CPU
caffe.set_mode_gpu(); % for GPU
caffe.set_device(gpu_id);

test_sequence_data_path = '/media/tjc/RSE/test/rse3/test_sequence_data/ldp/P_s10_3392x1984_crop_qp37_loopfilter/';  % this place should change!!!!!!!!!!!!2222222
test_sequence_label_path = '/media/tjc/RSE/test/rse3/test_sequence_label/s10_3392x1984_crop/';   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%333333333
dData = dir([test_sequence_data_path, '*.yuv']);   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dLabel = dir([test_sequence_label_path, '*.yuv']); %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if ~exist('./sequence_results','file')
    mkdir('./sequence_results');
end




%     weights_foreground = ['/home/jiaby/tjc/tjc/RSE/model_res3/frames_ai_res3_final/se_res3_qp37_ai_frames_Filter_iter_255000.caffemodel'];
% if you want to use loop function, set startNUM and endNUM, if
% startNUM == endNUM, this forloop just run once.(tjc, 2018/10/05)
%    weights_foreground =
%    ['/home/jiaby/tjc/tjc/RSE/model_res3/se_res3_qp37_ai_frames_Filter_iter_',num2str(epochNUM),'.caffemodel'];%
%
model_path = '/media/tjc/RSE/test/rse3/vrcnn_test';


HM_org_set = [];
SeFiter_set = [];
HM_psnr_set = [];
h1_psnr_set = [];

for iii = 1:1:length(dData)
    
    weights_ai = ['/media/tjc/RSE/model/ai/vrcnn_ai_qp37_frames_1/vrcnn_Filter_iter_12000.caffemodel'];
    weights_ldp = ['/media/tjc/RSE/model/ldp/vrcnn_ldp_frames/VRCNN_qp37_ldp_frame_Filter_iter_207000.caffemodel'];
%     weights_ldp = ['/media/tjc/RSE/model/ldp/VDSR/VDSR_qp37_ldp_foreground_Filter_iter_18000.caffemodel'];
    
    disp(['i = ' num2str(iii)]);
    cell = [];
    frewind(fid1);
    %% tjc hei and wid should change number!
    
    %         disp([label_set_path, dData(iii).name]);
    
    % flabel = fopen('./filter/testqp/BasketballDrill_832x480qp27l.yuv','rb');
    flabel = fopen([test_sequence_label_path, dLabel(iii).name],'rb');
    sName = dLabel(iii).name();
    sName = sName(1:end-4);
    disp(sName);
    fdata =  fopen([test_sequence_data_path, dData(iii).name],'rb');
    % fdata = fopen('./filter/testqp/B.0asketballDrill_832x480.yuv','rb');
    
    s1 = strsplit(dLabel(iii).name, {'_','x','.'});
    hei = str2double(cell2mat(s1(4))); % this is right, hei and width should be change place!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    wid = str2double(cell2mat(s1(5)));
    
    
    label_luminance = fread(flabel, [hei,wid], 'uint8=>uint8');
    data_luminance = fread(fdata, [hei,wid], 'uint8=>uint8');
    %         imshow(data_luminance');
    
    data_cb = fread(fdata, [hei/2,wid/2], 'uint8=>uint8');
    data_cr = fread(fdata, [hei/2,wid/2], 'uint8=>uint8');
    data_cb = data_cb';
    data_cr = data_cr';
    
    fclose(flabel);
    fclose(fdata);
    
    label_Y = im2double(label_luminance);
    data_Y = im2double(data_luminance); % passby the im2double function , the value will mapping to [0~1]. tjc
    data_Y = data_Y';
    data_y = double(data_Y);
    
    label_Y = label_Y';
    im_y_gnd = label_Y;
    
    %         imshow(data_y);
    
    %% adaptively spilt
    temp = hei;
    hei = wid;
    wid = temp;
    
    % decide patch numbers
    hei_patch = ceil(hei/(thresh_hei+rf));
    wid_patch = ceil(wid/(thresh_wid+rf));
    hei_stride = ceil(hei/hei_patch);
    wid_stride = ceil(wid/wid_patch);
    
    
    % extract each patch for inference
    im_y_h = [];
    
    for x = 1 : hei_stride : hei
        for y = 1 : wid_stride : wid
            % decide the length of hei and wid for each patch
            use_start_x = x;
            use_start_y = y;
            if x - rf > 1 % add border
                ext_start_x = x-rf;
                posext_start_x = rf+1;
            else
                ext_start_x = x;
                posext_start_x = 1;
            end
            if y-rf > 1
                ext_start_y = y-rf;
                posext_start_y = rf+1;
            else
                ext_start_y = y;
                posext_start_y = 1;
            end
            
            use_end_x = use_start_x+hei_stride-1;
            use_end_y = use_start_y+wid_stride-1;
            
            
            if use_start_x+hei_stride+rf-1 <= hei
                hei_length = hei_stride+rf;
                ext_end_x = use_start_x+hei_length-1;
                posext_end_x = hei_length-rf+posext_start_x-1;
                
            else
                hei_length = hei-ext_start_x+1;
                ext_end_x = ext_start_x+hei_length-1;
                posext_end_x = hei_length;
                use_end_x = ext_start_x+hei_length-1;
            end
            if use_start_y+wid_stride+rf-1 <= wid
                wid_length = wid_stride+rf;
                ext_end_y = use_start_y+wid_length-1;
                posext_end_y = wid_length-rf+posext_start_y-1;
                
            else
                wid_length = wid-ext_start_y+1;
                ext_end_y = ext_start_y+wid_length-1;
                posext_end_y = wid_length;
                use_end_y = ext_start_y+wid_length-1;
            end
            
            subim_input = data_y(ext_start_x : ext_end_x, ext_start_y : ext_end_y);  % input
            
            %                         imshow(subim_input);
            
            data = permute(subim_input,[2, 1, 3]);
            
            %         imshow(data);
            
            model = [model_path '.prototxt'];
            if iii == 1
                subim_output = do_cnn(model,weights_ai,data);
            else
                subim_output = do_cnn(model,weights_ldp,data);
            end;
            
            
            
            
            
            
            %         imshow(subim_output);
            
            subim_output = subim_output';
            %         imshow(subim_output);
            
            subim_output = subim_output(posext_start_x:posext_end_x,posext_start_y:posext_end_y);
            
            %         fill im_h with sub_output
            im_y_h(use_start_x:use_end_x,use_start_y:use_end_y) = subim_output;
            
            %         imshow(im_y_h);
            
        end
    end
    
    %         imshow(im_y_h);
    %         imshow(im_y_h2);
    
    %% remove border
    im_y_h1 = uint8(single(im_y_h) * 255); % rec pic
    
    im_y_gnd1 = uint8(single(im_y_gnd) * 255);  % This palce will make a big mistake;becareful.
    
    data_y = uint8(single(data_y) * 255);
    
    %             imshow(im_y_gnd1); % it can works! for save    time!
    
    
    
    %% attention: im_y_h1 is the final Y. 2018/10/7/ Tongjunchao-TJC
    while feof(fid1) == 0
        tline = fgetl(fid1);
        matche = strfind(tline, sName);
        if ~isempty(matche)
            Merge = fgetl(fid1);
            ss = strsplit(Merge, {'_'});
            num = str2double(cell2mat(ss(2)));
            for t = 1: num
                tline = fgetl(fid1);
                matTemp = strsplit(tline, {'  '});
                cell = [cell;matTemp];
            end;
            break;
        end;
    end
    
    %
    % The numbner of excel should add 1. due to index start from 0 of the
    % C++; (tjc)
    s = size(cell);
    %         disp(cell(2,3));
    
    %         disp(size(im_y_h1));
    h1_array = [];  % merge foreground area of fore_model.
    data_array = [];
    gnd_array = []; % merge foreground area of label.
    
    temp_end = wid;
    wid = hei;
    hei = temp_end;
    for k = 1 : s(1)
        txt_x1 = round(str2double(cell(k, 2))) + 1; 
        if txt_x1 > wid
            txt_x1 = wid;
        end;
        txt_y1 = round(str2double(cell(k, 1))) + 1;
        if txt_y1 > hei
            txt_y1 = hei;
        end;
        txt_x2 = round(str2double(cell(k, 4))) + 1;
        if txt_x2 > wid
            txt_x2 = wid;
        end;
        txt_y2 = round(str2double(cell(k, 3))) + 1;
        if txt_y2 > hei
            txt_y2 = hei;
        end;
        
        temp_area_h = im_y_h1(txt_x1 : txt_x2, txt_y1 : txt_y2);  % only get im_y_h1 \ data_y \ im_y_gnd1 Y number;
        temp_area_h = temp_area_h(:);
        h1_array = [h1_array; temp_area_h];
        
        temp_area_data = data_y(txt_x1 : txt_x2, txt_y1 : txt_y2);
        temp_area_data = temp_area_data(:);
        data_array = [data_array; temp_area_data];
        %             disp(h1_array);
        
        temp_area_gnd = im_y_gnd1(txt_x1 : txt_x2, txt_y1 : txt_y2);
        temp_area_gnd = temp_area_gnd(:);
        gnd_array = [gnd_array; temp_area_gnd];
        %             disp(gnd_array);
    end;
    
    h1_psnr = psnr_tjc(h1_array, gnd_array);
    h1_psnr_set = [h1_psnr_set; h1_psnr];
    
    data_psnr = psnr_tjc(data_array, gnd_array);
    HM_psnr_set = [HM_psnr_set; data_psnr];
    
    HM_org_psnr = compute_psnr(im_y_gnd1, data_y);
    HM_org_set = [HM_org_set, HM_org_psnr];
    
    %%
    %         imshow(im_y_h1);
    im_y = im_y_h1;
    %% save images (it can works) for save time!
    
    data_cb = imresize(data_cb, 2,'bicubic');
    data_cr = imresize(data_cr, 2,'bicubic');
    
    
    ycbcr = cat(3, (im_y), (data_cb), (data_cr));
    im_YUV = ycbcr2rgb(ycbcr);
    %         imshow(im_YUV);
    
    %     imwrite(im_h1,fullfile('./','rec1.png')); % maybe can run! yes, it can!
    
    %% compute PSNR and SSIM and IFC
    
    drrn(1) = compute_psnr(im_y_gnd1,im_y);
    
    %         imwrite(im_YUV, ['./sequence_results/', 'epoch:', num2str(NUM), '_', dData(iii).name(1:end-4), '_psnr:', num2str(drrn(1)), '.png']);
    
    drrn(2) = ssim_index(im_y_gnd1,im_y);
    
    %         drrn(3) = ifcvec(double(im_y_gnd1),double(im_y));
    
    SeFiter_set = [SeFiter_set; drrn];
end
% loop many YUVfiles.
%     disp(fore_psnr_set);
%     disp(sum(fore_psnr_set));
%     disp(length(fore_psnr_set));
%     disp(sum(fore_psnr_set)/length(fore_psnr_set));
mean_SeFilter = [mean_SeFilter; [mean(SeFiter_set(:,1)) mean(SeFiter_set(:,2))]]; %  mean(SeFiter_set(:,3))
mean_h1_psnr = [mean_h1_psnr; mean(h1_psnr_set(:, 1))];
mean_HM_psnr = mean(HM_psnr_set(:, 1));

gain = mean_h1_psnr(1, :) - mean_HM_psnr;

mean_HM_frame_psnr = mean(HM_org_set);


%%% save PSNR and SSIM metrics

tjc = 998;

disp(model_path);
disp(['epoch: ', num2str(tjc),'---- HM_org_psnr= ' num2str(mean_HM_frame_psnr)]);
disp(['epoch: ', num2str(tjc),'---- Frame_level :SEnet_Filter = ' num2str(mean_SeFilter(1,:))]);
disp(['epoch: ', num2str(tjc),'---- SEnet_Filter precise_localization_foreModel foreground PSNR= ' num2str(mean_h1_psnr(1, :))]);
disp(['epoch: ', num2str(tjc),'---- SEnet_Filter HM_traditional PSNR= ' num2str(mean_HM_psnr)]);
disp(['gain: ' num2str(gain) 'dB']);

