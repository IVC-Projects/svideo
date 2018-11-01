clear;close all;
setenv('LC_ALL','C');

%% settings !!!!!!!!!!!!need change 3 places!tjc 2018.6.25
folder_label = '/media/jiaby/Elements/ai/qp37/foreground/label/';  %!!!!!!!!!!!!!!!!!!!!! need change(tjc)1!!!!!!!!!!!!!!!!!!!!!!!!!!2018.6.22
folder_train = '/media/jiaby/Elements/ai/qp37/foreground/data/';  %/cross-validation/
size_input = 64;
size_label = 64;
stride = 64;
savepath = ['./abc' num2str(size_input) '_qp37.h5'];%!!!!!!!!!!!!!!!!!!! need change(tjc)2!!!!!!!!!!!!!!!!!!!!!!!!!!2018.6.2

%% initialization
data = zeros(size_input, size_input, 1, 1);
label = zeros(size_label, size_label,1, 1);
padding = abs(size_input - size_label)/2;
count = 0;

%% generate data

% disp([filepaths_label.name]);
% disp('----------------');
% disp([filepaths_train.name]);
% disp(filepaths_train(1).name);
% disp(class(filepaths_train(1).name));
tic;
%% sssssssssssssssssssssssssssssssssssssss

%% generate data
filepaths_label = dir(fullfile(folder_label,'*.yuv'));
filepaths_train = dir(fullfile(folder_train,'*.yuv'));
% disp([filepaths_label.name]);
% disp(class(filepaths_train(1).name));
tic;

fid = fopen('./merge.txt','rt');
% disp(length(filepaths_train));
for i = 1 : length(filepaths_label)
    cell = [];
    frewind(fid);
    disp(['i = ' num2str(i)]);
    
    s1 = strsplit(filepaths_label(i).name, {'_','x','.'});
    %     disp(class(s1(2)));
    hei_L = str2double(cell2mat(s1(4))); %
    wid_L = str2double(cell2mat(s1(5)));  %
    
    disp(['hight:',num2str(hei_L)]);
    disp(['width:',num2str(wid_L)]);
    
    hei_T = str2double(cell2mat(s1(4)));
    wid_T = str2double(cell2mat(s1(5)));
    %     disp(['hight:',num2str(hei_T)]);
    %     disp(['width:',num2str(wid_T)]);
    
    flabel = fopen(['./label/',filepaths_label(i).name],'rb');%!!!!!!!!!!!! need change(tjc)8!!!!!!!!!
    
    sName = filepaths_label(i).name();
    sName = sName(1:end-4);
    disp(sName);
    fdata = fopen(['./data/',filepaths_train(i).name],'rb');%!!!!!!!!!!!! need change(tjc)9!!!!!!!!!!!2018.6.2
    
    label_luminance = fread(flabel, [hei_L,wid_L], 'uint8=>uint8');
    data_luminance = fread(fdata, [hei_T,wid_T], 'uint8=>uint8');
    
    fclose(flabel);
    fclose(fdata);
    
    label_Y = im2double(label_luminance);
    data_Y = im2double(data_luminance);
    data_Y = data_Y';
    label_Y = label_Y';
    %    disp(label_Y);
%     imshow(data_Y);
%     
%     imshow(label_Y);
    %     disp(data_Y);
    %     disp(size(data_Y));
    
    %%  Read txt to get positions.
    while feof(fid) == 0
        tline = fgetl(fid);
        matche = strfind(tline, sName);
        if ~isempty(matche)
            Merge = fgetl(fid);
            %        disp(Merge);
            ss = strsplit(Merge, {'_'});
            num = str2double(cell2mat(ss(2)));
            for t = 1: num
                tline = fgetl(fid);
                matTemp = strsplit(tline, {'  '});
                cell = [cell;matTemp];
                
            end;
%             disp(round(str2double(cell(1, 2))));
%             disp(class(cell));
            %        temp = str2double();
%             disp(cell);
%             disp(class(cell));
%             disp(num);
            break;
        end;
    end
    
    %%
    % The numbner of excel should add 1. due to index start from 0 of the
    % C++; (tjc)
    s = size(cell);
    
    for x = 1 : stride : wid_T-size_input+1  %1344
        for y = 1 :stride : hei_T-size_input+1  % 1984
            flag = 0;
            subim_input = data_Y(x : x+size_input-1, y : y+size_input-1);
            subim_label = label_Y(x+padding : x+padding+size_label-1, y+padding : y+padding+size_label-1);
            
            mat_x1 = x;
%             disp(x);
            mat_y1 = y;
%             disp(y);
            mat_x2 = x + size_input - 1;
%             disp(mat_x2);
            mat_y2 = y + size_input - 1;
%             disp(mat_y2);

%             disp(['mat_y1 = ', num2str(mat_y1)]);
            
            for k = 1 : s(1)
                txt_x1 = round(str2double(cell(k, 2))) + 1;  %
                txt_y1 = round(str2double(cell(k, 1))) + 1;
                txt_x2 = round(str2double(cell(k, 4))) + 1;
                txt_y2 = round(str2double(cell(k, 3))) + 1;
                if ((max(mat_x1, txt_x1) < min(mat_x2, txt_x2)) && (max(mat_y1, txt_y1) < min(mat_y2, txt_y2)))
                    flag = 1;
                    break;
                end;
            end;
            
            if flag == 0
                continue;
            end;
            
            
%             imshow(subim_input);
%             imshow(subim_label);
%             imwrite(subim_label,['./foreground/', num2str(i), '_', num2str(y), '_', num2str(x), '.png']);
            count=count+1;
            data_temp = subim_input;
            label_temp = subim_label;
            
            data(:, :, :, count) = single(data_temp);
            
            %             imshow(data); % this operation may error!
            
            
            label(:, :, :, count) = single(label_temp);
            
            %             imshow(label);
            
        end
    end
end

fclose(fid); % close txt File which have positions.


%%
toc;
disp(['run time1: ',num2str(toc)]);

tic;
disp('start to randperm(count)');
order = randperm(count);
disp('start to data');
data = data(:, :, :, order);
disp('start to label');
label = label(:, :, :, order);

disp('start to writing to HDF5');

%% writing to HDF5
chunksz = 64;
created_flag = false;
totalct = 0;

for batchno = 1:floor(count/chunksz)
    last_read=(batchno-1)*chunksz;
    batchdata = data(:,:,:,last_read+1:last_read+chunksz);
    batchlabs = label(:,:,:,last_read+1:last_read+chunksz);
    
    startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
    curr_dat_sz = store2hdf5(savepath, batchdata, batchlabs, ~created_flag, startloc, chunksz);
    created_flag = true;
    totalct = curr_dat_sz(end);
end

toc;
disp(['run time2: ',num2str(toc)]);
h5disp(savepath);