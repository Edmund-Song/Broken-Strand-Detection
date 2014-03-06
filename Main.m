%% source code of the OGW broken strand detection method
%%make sure Libsvm and HoG have been correctly installed.

function [Label]=Main(NumF)
%% return the type of environment state: normal wire(1), broken wire(2) and
%% obstacles(3) 

% SVM traning
[SVMmodel_Wire_Nonwire SVMmodel_Brokenwire_NormalWire]=SVM_Model_Train;

Detection_Image_No=NumF;

% ROI selection
[I_ROI]= ROI_Selection(Detection_Image_No);

% HoG extraction Leo (2012, Aug). Histograms of Oriented Gradients. Available at. http://www.mathworks.com/matlabcentral/fileexchange/33863-histograms-of-oriented-gradients
Detection_Img_HoG_Vec=HoG(double(I_ROI));

% classification
[label_1, ~, ~] = svmpredict(1, Detection_Img_HoG_Vec',SVMmodel_Wire_Nonwire)

if (label_1==-1)
    Label = 3;%Obstacles 
else
    [label_2, ~, ~] = svmpredict(1, Detection_Img_HoG_Vec',SVMmodel_Brokenwire_NormalWire);
    if (label_2==-1)
        Label = 2;%Broken Wire
    else
        Label= 1;%Normal Wire
    end
end


function [SVMmodel_Wire_Nonwire SVMmodel_Brokenwire_NormalWire]=SVM_Model_Train
%% return two trained SVM classifiers: classifier for wires and obstacles and classifier for normal wire and broken wire
% C.C. Chang and C.J. Lin, ¡°LIBSVM : a library for support vector machines,¡± ACM Transactions on Intelligent Systems and Technology,  2:27:1--27:27, 2011. Software available at http://www.csie.ntu.edu.tw/~cjlin/libsvm

% import the training data
TrainingIns_Wire_Nonwire = importdata('TrainingIns_Wire_Nonwire.mat');
TrainingLabel_Wire_Nonwire = importdata('TrainingLabel_Wire_Nonwire.mat');
TrainingIns_Brokenwire_NormalWire = importdata('TrainingIns_Brokenwire_NormalWire.mat');
TrainingLabel_Brokenwire_NormalWire = importdata('TrainingLabel_Brokenwire_NormalWire.mat');

% SVM training
SVMmodel_Wire_Nonwire=svmtrain(TrainingLabel_Wire_Nonwire,TrainingIns_Wire_Nonwire) ;
SVMmodel_Brokenwire_NormalWire=svmtrain(TrainingLabel_Brokenwire_NormalWire,TrainingIns_Brokenwire_NormalWire,'-s 0 -t 2 -g 0.5 -wi 5');



function [I_ROI]= ROI_Selection(Image_Num)
 
%%%%% in the image Filename to cut a area which we are interested in. Image_Num
%%%%% can indicate the image to be processed. 
 
strtemp=strcat('E:\PowerDliveryCode\Frames\p',int2str(Image_Num),'.png'); %%% set the directory and name of the detection image
I0 = imread (strtemp); % read the image
[Height,Width,c]= size(I0);%image size
 
%%%%image preprocessing
I_Crop=imcrop(I0,[round(0) round(0.05*Height) round(Width) round(0.3*Height)]);%Image cropping
I_Crop_Gray=rgb2gray(I_Crop);% Turn into gray image
I_Crop_Gray_HE=imadjust(I_Crop_Gray,[0 0.5],[]);%Histogram Equalization
I_Edge=edge(I_Crop_Gray_HE,'canny',0.05);
 
%%Hough Transform, find OGW boundaries by straight line detection.
[H,T,R]=hough(I_Edge,'RhoResolution',0.1,'Theta',-10:0.05:10);%% angle scope  from -10 to 10 
 
P=houghpeaks(H,50,'threshold',ceil(0.1*max(H(:)))); % find first 50 lines whose intensity is higher than 0.1*peaks
 
lines=houghlines(I_Edge,T,R,P,'FillGap',5,'MinLength',10);  %abandon the lines less than 10 and fill the gaps less than 5
 
%%% restore information of the extracted straight lines to Line_Infro
for k=1:length(lines) 
    xy=[lines(k).point1;lines(k).point2];
    Line_Infro(k,:) =  [lines(k).point1 lines(k).point2];
end
 
 
%%% Find OGW boundaries
OGW_Boundary_L=-1;
OGW_Boundary_R=-1;
 
    for kt = 1:length(lines)
    C1=kt;
    for j = (1+kt):length(lines)
        C2=j;
        if (abs(Line_Infro(C1,1)+Line_Infro(C2,1)+Line_Infro(C1,3)+Line_Infro(C2,3)-1440)<30)& (abs(abs(Line_Infro(C1,1)-Line_Infro(C2,1))+abs(Line_Infro(C1,3)-Line_Infro(C2,3))-100)<30) 
            % two conditions 1. in the middle of the image 2. OGW width fixed
            OGW_Boundary_L=(Line_Infro(C1,1)+Line_Infro(C1,3))/2;
            OGW_Boundary_R=(Line_Infro(C2,1)+Line_Infro(C2,3))/2;
        end
    end
    end
 
    Initial_ROI_Location=[round(11*Width/24) round(0.05*Height) 72 56]; % define a artifical ROI if OGW is not found (in the case of obstacles)  
     
if (OGW_Boundary_L>0) & (OGW_Boundary_L>0) %in the case OGW found
    ROI_Location= [ (OGW_Boundary_L-10) round(0.05*Height) 72 56];% define ROI area
    I_ROI=imcrop(I_Crop_Gray_HE,Initial_ROI_Location);
    ROI_M=1;% to indicate if OGW is found
 elseif isstruct(lines)==0|length(lines)<2 % there is no lines in the image. 
    I_ROI=imcrop(I_Crop_Gray_HE,Initial_ROI_Location);
    ROI_M=0;
else
    I_ROI=imcrop(I_Crop_Gray_HE,Initial_ROI_Location);% All extracted lines are not OGW boundaries
    ROI_M=-1; 
end

%% HoG extraction	Leo (2012, Aug). Histograms of Oriented Gradients. Available at. http://www.mathworks.com/matlabcentral/fileexchange/33863-histograms-of-oriented-gradients
% Copyright (c) 2011, Leo



% #include <stdlib.h>
% #include <math.h>
% #include <mex.h>
% #include <vector>
% 
% using namespace std;   
% 
% void HoG(double *pixels, double *params, int *img_size, double *dth_des, unsigned int grayscale){
%     
%     const float pi = 3.1415926536;
%     
%     int nb_bins       = (int) params[0];
%     double cwidth     =  params[1];
%     int block_size    = (int) params[2];
%     int orient        = (int) params[3];
%     double clip_val   = params[4];
%     
%     int img_width  = img_size[1];
%     int img_height = img_size[0];
%     
%     int hist1= 2+ceil(-0.5 + img_height/cwidth);
%     int hist2= 2+ceil(-0.5 + img_width/cwidth);
%     
%     double bin_size = (1+(orient==1))*pi/nb_bins;
%     
%     float dx[3], dy[3], grad_or, grad_mag, temp_mag;
%     float Xc, Yc, Oc, block_norm;
%     int x1, x2, y1, y2, bin1, bin2;
%     int des_indx = 0;
%     
%     vector<vector<vector<double> > > h(hist1, vector<vector<double> > (hist2, vector<double> (nb_bins, 0.0) ) );    
%     vector<vector<vector<double> > > block(block_size, vector<vector<double> > (block_size, vector<double> (nb_bins, 0.0) ) );
%     
%     //Calculate gradients (zero padding)
%     
%     for(unsigned int y=0; y<img_height; y++) {
%         for(unsigned int x=0; x<img_width; x++) {
%             if (grayscale == 1){
%                 if(x==0) dx[0] = pixels[y +(x+1)*img_height];
%                 else{
%                     if (x==img_width-1) dx[0] = -pixels[y + (x-1)*img_height];
%                     else dx[0] = pixels[y+(x+1)*img_height] - pixels[y + (x-1)*img_height];
%                 }
%                 if(y==0) dy[0] = -pixels[y+1+x*img_height];
%                 else{
%                     if (y==img_height-1) dy[0] = pixels[y-1+x*img_height];
%                     else dy[0] = -pixels[y+1+x*img_height] + pixels[y-1+x*img_height];
%                 }
%             }
%             else{
%                 if(x==0){
%                     dx[0] = pixels[y +(x+1)*img_height];
%                     dx[1] = pixels[y +(x+1)*img_height + img_height*img_width];
%                     dx[2] = pixels[y +(x+1)*img_height + 2*img_height*img_width];                    
%                 }
%                 else{
%                     if (x==img_width-1){
%                         dx[0] = -pixels[y + (x-1)*img_height];                        
%                         dx[1] = -pixels[y + (x-1)*img_height + img_height*img_width];
%                         dx[2] = -pixels[y + (x-1)*img_height + 2*img_height*img_width];
%                     }
%                     else{
%                         dx[0] = pixels[y+(x+1)*img_height] - pixels[y + (x-1)*img_height];
%                         dx[1] = pixels[y+(x+1)*img_height + img_height*img_width] - pixels[y + (x-1)*img_height + img_height*img_width];
%                         dx[2] = pixels[y+(x+1)*img_height + 2*img_height*img_width] - pixels[y + (x-1)*img_height + 2*img_height*img_width];
%                         
%                     }
%                 }
%                 if(y==0){
%                     dy[0] = -pixels[y+1+x*img_height];
%                     dy[1] = -pixels[y+1+x*img_height + img_height*img_width];
%                     dy[2] = -pixels[y+1+x*img_height + 2*img_height*img_width];
%                 }
%                 else{
%                     if (y==img_height-1){
%                         dy[0] = pixels[y-1+x*img_height];
%                         dy[1] = pixels[y-1+x*img_height + img_height*img_width];
%                         dy[2] = pixels[y-1+x*img_height + 2*img_height*img_width];
%                     }
%                     else{
%                         dy[0] = -pixels[y+1+x*img_height] + pixels[y-1+x*img_height];
%                         dy[1] = -pixels[y+1+x*img_height + img_height*img_width] + pixels[y-1+x*img_height + img_height*img_width];
%                         dy[2] = -pixels[y+1+x*img_height + 2*img_height*img_width] + pixels[y-1+x*img_height + 2*img_height*img_width];
%                     }
%                 }
%             }
%             
%             grad_mag = sqrt(dx[0]*dx[0] + dy[0]*dy[0]);
%             grad_or= atan2(dy[0], dx[0]);
%             
%             if (grayscale == 0){
%                 temp_mag = grad_mag;
%                 for (unsigned int cli=1;cli<3;++cli){
%                     temp_mag= sqrt(dx[cli]*dx[cli] + dy[cli]*dy[cli]);
%                     if (temp_mag>grad_mag){
%                         grad_mag=temp_mag;
%                         grad_or= atan2(dy[cli], dx[cli]);
%                     }
%                 }
%             }
%             
%             if (grad_or<0) grad_or+=pi + (orient==1) * pi;
% 
%             // trilinear interpolation
%             
%             bin1 = (int)floor(0.5 + grad_or/bin_size) - 1;
%             bin2 = bin1 + 1;
%             x1   = (int)floor(0.5+ x/cwidth);
%             x2   = x1+1;
%             y1   = (int)floor(0.5+ y/cwidth);
%             y2   = y1 + 1;
%             
%             Xc = (x1+1-1.5)*cwidth + 0.5;
%             Yc = (y1+1-1.5)*cwidth + 0.5;
%             
%             Oc = (bin1+1+1-1.5)*bin_size;
%             
%             if (bin2==nb_bins){
%                 bin2=0;
%             }
%             if (bin1<0){
%                 bin1=nb_bins-1;
%             }            
%            
%             h[y1][x1][bin1]= h[y1][x1][bin1]+grad_mag*(1-((x+1-Xc)/cwidth))*(1-((y+1-Yc)/cwidth))*(1-((grad_or-Oc)/bin_size));
%             h[y1][x1][bin2]= h[y1][x1][bin2]+grad_mag*(1-((x+1-Xc)/cwidth))*(1-((y+1-Yc)/cwidth))*(((grad_or-Oc)/bin_size));
%             h[y2][x1][bin1]= h[y2][x1][bin1]+grad_mag*(1-((x+1-Xc)/cwidth))*(((y+1-Yc)/cwidth))*(1-((grad_or-Oc)/bin_size));
%             h[y2][x1][bin2]= h[y2][x1][bin2]+grad_mag*(1-((x+1-Xc)/cwidth))*(((y+1-Yc)/cwidth))*(((grad_or-Oc)/bin_size));
%             h[y1][x2][bin1]= h[y1][x2][bin1]+grad_mag*(((x+1-Xc)/cwidth))*(1-((y+1-Yc)/cwidth))*(1-((grad_or-Oc)/bin_size));
%             h[y1][x2][bin2]= h[y1][x2][bin2]+grad_mag*(((x+1-Xc)/cwidth))*(1-((y+1-Yc)/cwidth))*(((grad_or-Oc)/bin_size));
%             h[y2][x2][bin1]= h[y2][x2][bin1]+grad_mag*(((x+1-Xc)/cwidth))*(((y+1-Yc)/cwidth))*(1-((grad_or-Oc)/bin_size));
%             h[y2][x2][bin2]= h[y2][x2][bin2]+grad_mag*(((x+1-Xc)/cwidth))*(((y+1-Yc)/cwidth))*(((grad_or-Oc)/bin_size));
%         }
%     }
%     
%     
%     
%     //Block normalization
%     
%     for(unsigned int x=1; x<hist2-block_size; x++){
%         for (unsigned int y=1; y<hist1-block_size; y++){
%             
%             block_norm=0;
%             for (unsigned int i=0; i<block_size; i++){
%                 for(unsigned int j=0; j<block_size; j++){
%                     for(unsigned int k=0; k<nb_bins; k++){
%                         block_norm+=h[y+i][x+j][k]*h[y+i][x+j][k];
%                     }
%                 }
%             }
%             
%             block_norm=sqrt(block_norm);
%             for (unsigned int i=0; i<block_size; i++){
%                 for(unsigned int j=0; j<block_size; j++){
%                     for(unsigned int k=0; k<nb_bins; k++){
%                         if (block_norm>0){
%                             block[i][j][k]=h[y+i][x+j][k]/block_norm;
%                             if (block[i][j][k]>clip_val) block[i][j][k]=clip_val;
%                         }
%                     }
%                 }
%             }
%             
%             block_norm=0;
%             for (unsigned int i=0; i<block_size; i++){
%                 for(unsigned int j=0; j<block_size; j++){
%                     for(unsigned int k=0; k<nb_bins; k++){
%                         block_norm+=block[i][j][k]*block[i][j][k];
%                     }
%                 }
%             }
%             
%             block_norm=sqrt(block_norm);
%             for (unsigned int i=0; i<block_size; i++){
%                 for(unsigned int j=0; j<block_size; j++){
%                     for(unsigned int k=0; k<nb_bins; k++){
%                         if (block_norm>0) dth_des[des_indx]=block[i][j][k]/block_norm;
%                         else dth_des[des_indx]=0.0;
%                         des_indx++;
%                     }
%                 }
%             }
%         }
%     }
% }
% 
% 
% void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
%     
%     double *pixels, *dth_des, *params;
%     int nb_bins, block_size;
%     int img_size[2];
%     unsigned int grayscale = 1;
%     
%     if (nlhs>1)  mexErrMsgTxt("Too many output arguments");
%     if (nrhs==0) mexErrMsgTxt("No Image -> No HoG");
%     
%     if (mxGetClassID(prhs[0])!=6) mexErrMsgTxt("Matrix is not of type double");
%     
%     pixels     = mxGetPr(prhs[0]);    
%     
%     img_size[0] = mxGetM(prhs[0]);
%     img_size[1]  = mxGetN(prhs[0]);
%     if (mxGetNumberOfDimensions(prhs[0])==3){
%         img_size[1] /= 3;
%         grayscale = 0;
%     }
%     
%     if (nrhs>1){
%         params     = mxGetPr(prhs[1]);
%         if (params[0]<=0) mexErrMsgTxt("Number of orientation bins must be positive");
%         if (params[1]<=0) mexErrMsgTxt("Cell size must be positive");
%         if (params[2]<=0) mexErrMsgTxt("Block size must be positive");
%     }
%     else {
function [Label]=Main(NumF)
%% return the type of environment state: normal wire(1), broken wire(2) and
%% obstacles (3) 
 
% SVM traning
[SVMmodel_Wire_Nonwire SVMmodel_Brokenwire_NormalWire]=SVM_Model_Train;
 
Detection_Image_No=NumF;
 
% ROI selection
[I_ROI]= ROI_Selection(Detection_Image_No);
 
% HoG extraction Leo (2012, Aug). Histograms of Oriented Gradients. Available at. http://www.mathworks.com/matlabcentral/fileexchange/33863-histograms-of-oriented-gradients
Detection_Img_HoG_Vec=HoG(double(I_ROI));
 
% classification
[label_1, ~, ~] = svmpredict(1, Detection_Img_HoG_Vec',SVMmodel_Wire_Nonwire)
 
if (label_1==-1)
    Label = 3;%Obstacles 
else
    [label_2, ~, ~] = svmpredict(1, Detection_Img_HoG_Vec',SVMmodel_Brokenwire_NormalWire);
    if (label_2==-1)
        Label = 2;%Broken Wire
    else
        Label= 1;%Normal Wire
    end
end
 
 
function [SVMmodel_Wire_Nonwire SVMmodel_Brokenwire_NormalWire]=SVM_Model_Train
%% return two trained SVM classifiers: classifier for wires and obstacles and classifier for normal wire and broken wire
% C.C. Chang and C.J. Lin, ?¡ãLIBSVM : a library for support vector machines,?¡À ACM Transactions on Intelligent Systems and Technology,  2:27:1--27:27, 2011. Software available at http://www.csie.ntu.edu.tw/~cjlin/libsvm
 
% import the training data
TrainingIns_Wire_Nonwire = importdata('TrainingIns_Wire_Nonwire.mat');
TrainingLabel_Wire_Nonwire = importdata('TrainingLabel_Wire_Nonwire.mat');
TrainingIns_Brokenwire_NormalWire = importdata('TrainingIns_Brokenwire_NormalWire.mat');
TrainingLabel_Brokenwire_NormalWire = importdata('TrainingLabel_Brokenwire_NormalWire.mat');
 
% SVM training
SVMmodel_Wire_Nonwire=svmtrain(TrainingLabel_Wire_Nonwire,TrainingIns_Wire_Nonwire) ;
SVMmodel_Brokenwire_NormalWire=svmtrain(TrainingLabel_Brokenwire_NormalWire,TrainingIns_Brokenwire_NormalWire,'-s 0 -t 2 -g 0.5 -wi 5');
 
 
 
function [I_ROI]= ROI_Selection(Image_Num)
 
%%%%% get a region in the image which we are interested in. Image_Num
%%%%% is the code number of the image to be processed. 
 
strtemp=strcat('E:\PowerDliveryCode\Frames\p',int2str(Image_Num),'.png'); %%% set the directory and name of the image to be processed
I0 = imread (strtemp); % read the image
[Height,Width,c]= size(I0);%image size
 
%%%%image preprocessing
I_Crop=imcrop(I0,[round(0) round(0.05*Height) round(Width) round(0.3*Height)]);%Image cropping
I_Crop_Gray=rgb2gray(I_Crop);% Turn into gray image
I_Crop_Gray_HE=imadjust(I_Crop_Gray,[0 0.5],[]);%Histogram Equalization
I_Edge=edge(I_Crop_Gray_HE,'canny',0.05);
 
%%Hough Transform, find OGW boundaries by straight line detection.
[H,T,R]=hough(I_Edge,'RhoResolution',0.1,'Theta',-10:0.05:10);%% angle scope  from -10 to 10 
 
P=houghpeaks(H,50,'threshold',ceil(0.1*max(H(:)))); % find first 50 lines whose intensity is higher than 0.1*peaks
 
lines=houghlines(I_Edge,T,R,P,'FillGap',5,'MinLength',10);  %abandon the lines less than 10 and fill the gaps less than 5
 
%%% restore information of the extracted straight lines to Line_Infro
for k=1:length(lines) 
    xy=[lines(k).point1;lines(k).point2];
    Line_Infro(k,:) =  [lines(k).point1 lines(k).point2];
end
 
 
%%% Find OGW boundaries
OGW_Boundary_L=-1;
OGW_Boundary_R=-1;
 
    for kt = 1:length(lines)
    C1=kt;
    for j = (1+kt):length(lines)
        C2=j;
        if (abs(Line_Infro(C1,1)+Line_Infro(C2,1)+Line_Infro(C1,3)+Line_Infro(C2,3)-1440)<30)& (abs(abs(Line_Infro(C1,1)-Line_Infro(C2,1))+abs(Line_Infro(C1,3)-Line_Infro(C2,3))-100)<30) 
            % two conditions 1. in the middle of the image 2. OGW width fixed
            OGW_Boundary_L=(Line_Infro(C1,1)+Line_Infro(C1,3))/2;
            OGW_Boundary_R=(Line_Infro(C2,1)+Line_Infro(C2,3))/2;
        end
    end
    end
 
    Initial_ROI_Location=[round(11*Width/24) round(0.05*Height) 72 56]; % define a artifical ROI if OGW is not found (in the case of obstacles)  
     
if (OGW_Boundary_L>0) & (OGW_Boundary_L>0) %in the case OGW found
    ROI_Location= [ (OGW_Boundary_L-10) round(0.05*Height) 72 56];% define ROI area
    I_ROI=imcrop(I_Crop_Gray_HE,Initial_ROI_Location);
    ROI_M=1;% to indicate if OGW is found
 elseif isstruct(lines)==0|length(lines)<2 % there is no lines in the image. 
    I_ROI=imcrop(I_Crop_Gray_HE,Initial_ROI_Location);
    ROI_M=0;
else
    I_ROI=imcrop(I_Crop_Gray_HE,Initial_ROI_Location);% All extracted lines are not OGW boundaries
    ROI_M=-1; 
end
 
%% HoG extraction   Leo (2012, Aug). Histograms of Oriented Gradients. Available at. http://www.mathworks.com/matlabcentral/fileexchange/33863-histograms-of-oriented-gradients
% Copyright (c) 2011, Leo
 
 
 
% #include <stdlib.h>
% #include <math.h>
% #include <mex.h>
% #include <vector>
% 
% using namespace std;   
% 
% void HoG(double *pixels, double *params, int *img_size, double *dth_des, unsigned int grayscale){
%     
%     const float pi = 3.1415926536;
%     
%     int nb_bins       = (int) params[0];
%     double cwidth     =  params[1];
%     int block_size    = (int) params[2];
%     int orient        = (int) params[3];
%     double clip_val   = params[4];
%     
%     int img_width  = img_size[1];
%     int img_height = img_size[0];
%     
%     int hist1= 2+ceil(-0.5 + img_height/cwidth);
%     int hist2= 2+ceil(-0.5 + img_width/cwidth);
%     
%     double bin_size = (1+(orient==1))*pi/nb_bins;
%     
%     float dx[3], dy[3], grad_or, grad_mag, temp_mag;
%     float Xc, Yc, Oc, block_norm;
%     int x1, x2, y1, y2, bin1, bin2;
%     int des_indx = 0;
%     
%     vector<vector<vector<double> > > h(hist1, vector<vector<double> > (hist2, vector<double> (nb_bins, 0.0) ) );    
%     vector<vector<vector<double> > > block(block_size, vector<vector<double> > (block_size, vector<double> (nb_bins, 0.0) ) );
%     
%     //Calculate gradients (zero padding)
%     
%     for(unsigned int y=0; y<img_height; y++) {
%         for(unsigned int x=0; x<img_width; x++) {
%             if (grayscale == 1){
%                 if(x==0) dx[0] = pixels[y +(x+1)*img_height];
%                 else{
%                     if (x==img_width-1) dx[0] = -pixels[y + (x-1)*img_height];
%                     else dx[0] = pixels[y+(x+1)*img_height] - pixels[y + (x-1)*img_height];
%                 }
%                 if(y==0) dy[0] = -pixels[y+1+x*img_height];
%                 else{
%                     if (y==img_height-1) dy[0] = pixels[y-1+x*img_height];
%                     else dy[0] = -pixels[y+1+x*img_height] + pixels[y-1+x*img_height];
%                 }
%             }
%             else{
%                 if(x==0){
%                     dx[0] = pixels[y +(x+1)*img_height];
%                     dx[1] = pixels[y +(x+1)*img_height + img_height*img_width];
%                     dx[2] = pixels[y +(x+1)*img_height + 2*img_height*img_width];                    
%                 }
%                 else{
%                     if (x==img_width-1){
%                         dx[0] = -pixels[y + (x-1)*img_height];                        
%                         dx[1] = -pixels[y + (x-1)*img_height + img_height*img_width];
%                         dx[2] = -pixels[y + (x-1)*img_height + 2*img_height*img_width];
%                     }
%                     else{
%                         dx[0] = pixels[y+(x+1)*img_height] - pixels[y + (x-1)*img_height];
%                         dx[1] = pixels[y+(x+1)*img_height + img_height*img_width] - pixels[y + (x-1)*img_height + img_height*img_width];
%                         dx[2] = pixels[y+(x+1)*img_height + 2*img_height*img_width] - pixels[y + (x-1)*img_height + 2*img_height*img_width];
%                         
%                     }
%                 }
%                 if(y==0){
%                     dy[0] = -pixels[y+1+x*img_height];
%                     dy[1] = -pixels[y+1+x*img_height + img_height*img_width];
%                     dy[2] = -pixels[y+1+x*img_height + 2*img_height*img_width];
%                 }
%                 else{
%                     if (y==img_height-1){
%                         dy[0] = pixels[y-1+x*img_height];
%                         dy[1] = pixels[y-1+x*img_height + img_height*img_width];
%                         dy[2] = pixels[y-1+x*img_height + 2*img_height*img_width];
%                     }
%                     else{
%                         dy[0] = -pixels[y+1+x*img_height] + pixels[y-1+x*img_height];
%                         dy[1] = -pixels[y+1+x*img_height + img_height*img_width] + pixels[y-1+x*img_height + img_height*img_width];
%                         dy[2] = -pixels[y+1+x*img_height + 2*img_height*img_width] + pixels[y-1+x*img_height + 2*img_height*img_width];
%                     }
%                 }
%             }
%             
%             grad_mag = sqrt(dx[0]*dx[0] + dy[0]*dy[0]);
%             grad_or= atan2(dy[0], dx[0]);
%             
%             if (grayscale == 0){
%                 temp_mag = grad_mag;
%                 for (unsigned int cli=1;cli<3;++cli){
%                     temp_mag= sqrt(dx[cli]*dx[cli] + dy[cli]*dy[cli]);
%                     if (temp_mag>grad_mag){
%                         grad_mag=temp_mag;
%                         grad_or= atan2(dy[cli], dx[cli]);
%                     }
%                 }
%             }
%             
%             if (grad_or<0) grad_or+=pi + (orient==1) * pi;
% 
%             // trilinear interpolation
%             
%             bin1 = (int)floor(0.5 + grad_or/bin_size) - 1;
%             bin2 = bin1 + 1;
%             x1   = (int)floor(0.5+ x/cwidth);
%             x2   = x1+1;
%             y1   = (int)floor(0.5+ y/cwidth);
%             y2   = y1 + 1;
%             
%             Xc = (x1+1-1.5)*cwidth + 0.5;
%             Yc = (y1+1-1.5)*cwidth + 0.5;
%             
%             Oc = (bin1+1+1-1.5)*bin_size;
%             
%             if (bin2==nb_bins){
%                 bin2=0;
%             }
%             if (bin1<0){
%                 bin1=nb_bins-1;
%             }            
%            
%             h[y1][x1][bin1]= h[y1][x1][bin1]+grad_mag*(1-((x+1-Xc)/cwidth))*(1-((y+1-Yc)/cwidth))*(1-((grad_or-Oc)/bin_size));
%             h[y1][x1][bin2]= h[y1][x1][bin2]+grad_mag*(1-((x+1-Xc)/cwidth))*(1-((y+1-Yc)/cwidth))*(((grad_or-Oc)/bin_size));
%             h[y2][x1][bin1]= h[y2][x1][bin1]+grad_mag*(1-((x+1-Xc)/cwidth))*(((y+1-Yc)/cwidth))*(1-((grad_or-Oc)/bin_size));
%             h[y2][x1][bin2]= h[y2][x1][bin2]+grad_mag*(1-((x+1-Xc)/cwidth))*(((y+1-Yc)/cwidth))*(((grad_or-Oc)/bin_size));
%             h[y1][x2][bin1]= h[y1][x2][bin1]+grad_mag*(((x+1-Xc)/cwidth))*(1-((y+1-Yc)/cwidth))*(1-((grad_or-Oc)/bin_size));
%             h[y1][x2][bin2]= h[y1][x2][bin2]+grad_mag*(((x+1-Xc)/cwidth))*(1-((y+1-Yc)/cwidth))*(((grad_or-Oc)/bin_size));
%             h[y2][x2][bin1]= h[y2][x2][bin1]+grad_mag*(((x+1-Xc)/cwidth))*(((y+1-Yc)/cwidth))*(1-((grad_or-Oc)/bin_size));
%             h[y2][x2][bin2]= h[y2][x2][bin2]+grad_mag*(((x+1-Xc)/cwidth))*(((y+1-Yc)/cwidth))*(((grad_or-Oc)/bin_size));
%         }
%     }
%     
%     
%     
%     //Block normalization
%     
%     for(unsigned int x=1; x<hist2-block_size; x++){
%         for (unsigned int y=1; y<hist1-block_size; y++){
%             
%             block_norm=0;
%             for (unsigned int i=0; i<block_size; i++){
%                 for(unsigned int j=0; j<block_size; j++){
%                     for(unsigned int k=0; k<nb_bins; k++){
%                         block_norm+=h[y+i][x+j][k]*h[y+i][x+j][k];
%                     }
%                 }
%             }
%             
%             block_norm=sqrt(block_norm);
%             for (unsigned int i=0; i<block_size; i++){
%                 for(unsigned int j=0; j<block_size; j++){
%                     for(unsigned int k=0; k<nb_bins; k++){
%                         if (block_norm>0){
%                             block[i][j][k]=h[y+i][x+j][k]/block_norm;
%                             if (block[i][j][k]>clip_val) block[i][j][k]=clip_val;
%                         }
%                     }
%                 }
%             }
%             
%             block_norm=0;
%             for (unsigned int i=0; i<block_size; i++){
%                 for(unsigned int j=0; j<block_size; j++){
%                     for(unsigned int k=0; k<nb_bins; k++){
%                         block_norm+=block[i][j][k]*block[i][j][k];
%                     }
%                 }
%             }
%             
%             block_norm=sqrt(block_norm);
%             for (unsigned int i=0; i<block_size; i++){
%                 for(unsigned int j=0; j<block_size; j++){
%                     for(unsigned int k=0; k<nb_bins; k++){
%                         if (block_norm>0) dth_des[des_indx]=block[i][j][k]/block_norm;
%                         else dth_des[des_indx]=0.0;
%                         des_indx++;
%                     }
%                 }
%             }
%         }
%     }
% }
% 
% 
% void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
%     
%     double *pixels, *dth_des, *params;
%     int nb_bins, block_size;
%     int img_size[2];
%     unsigned int grayscale = 1;
%     
%     if (nlhs>1)  mexErrMsgTxt("Too many output arguments");
%     if (nrhs==0) mexErrMsgTxt("No Image -> No HoG");
%     
%     if (mxGetClassID(prhs[0])!=6) mexErrMsgTxt("Matrix is not of type double");
%     
%     pixels     = mxGetPr(prhs[0]);    
%     
%     img_size[0] = mxGetM(prhs[0]);
%     img_size[1]  = mxGetN(prhs[0]);
%     if (mxGetNumberOfDimensions(prhs[0])==3){
%         img_size[1] /= 3;
%         grayscale = 0;
%     }
%     
%     if (nrhs>1){
%         params     = mxGetPr(prhs[1]);
%         if (params[0]<=0) mexErrMsgTxt("Number of orientation bins must be positive");
%         if (params[1]<=0) mexErrMsgTxt("Cell size must be positive");
%         if (params[2]<=0) mexErrMsgTxt("Block size must be positive");
%     }
%     else {
%         params = new double[5];
%         params[0]=9;
%         params[1]=8;
%         params[2]=2;
%         params[3]=0;
%         params[4]=0.2;
%     }
%     
%     nb_bins       = (int) params[0];    
%     block_size    = (int) params[2];     
%     
%     int hist1= 2+ceil(-0.5 + img_size[0]/params[1]);
%     int hist2= 2+ceil(-0.5 + img_size[1]/params[1]);
% 
%     plhs[0] = mxCreateDoubleMatrix((hist1-2-(block_size-1))*(hist2-2-(block_size-1))*nb_bins*block_size*block_size, 1, mxREAL);
%     dth_des = mxGetPr(plhs[0]);
%     
%     HoG(pixels, params, img_size, dth_des, grayscale);
%     if (nrhs==1) delete[] params;
% }



%         params = new double[5];
%         params[0]=9;
%         params[1]=8;
%         params[2]=2;
%         params[3]=0;
%         params[4]=0.2;
%     }
%     
%     nb_bins       = (int) params[0];    
%     block_size    = (int) params[2];     
%     
%     int hist1= 2+ceil(-0.5 + img_size[0]/params[1]);
%     int hist2= 2+ceil(-0.5 + img_size[1]/params[1]);
% 
%     plhs[0] = mxCreateDoubleMatrix((hist1-2-(block_size-1))*(hist2-2-(block_size-1))*nb_bins*block_size*block_size, 1, mxREAL);
%     dth_des = mxGetPr(plhs[0]);
%     
%     HoG(pixels, params, img_size, dth_des, grayscale);
%     if (nrhs==1) delete[] params;
% }
