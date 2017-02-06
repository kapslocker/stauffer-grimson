#include <iostream>
#include <dirent.h>                 // for detecting frames
#include <vector>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;


/**
*  @params dir, frames
*  Fill vector frames with file names of images from directory dir.
*/
void getframes(string dir, vector<string> &frames){
  DIR *dp;
  struct dirent *dirp;
  dp = opendir(dir.c_str());
  while( (dirp = readdir(dp)) !=NULL){
    frames.push_back(string(dirp->d_name));
  }
  sort(frames.begin(),frames.end());
  closedir(dp);
  return;
}
/*
* Contains parameters for the program
*/
class Params{
    int width;
    int height;
    int K;
    float alpha, T;
  public:
    void initParams(int w, int h){
      width = w;
      height = h;
      K = 3;
      alpha = 0.3f;
      T = 0.75f;
    }
    int getCols(){
      return width;
    }
    int getRows(){
      return height;
    }
    int maxModes(){
      return K;
    }
}params;


vector<vector<vector<Vec3b> > > mu,covar;                                       // Mean and Covariance for each Gaussian at pixel (x,y).
vector<vector<vector<Vec3b> > > pr;                                               // Each Gaussian's contribution to the mixture at pixel (x,y)

void initialiseVec(){

  for(int i=0;i<params.getRows();i++){
    mu.push_back(vector<vector<Vec3b> >());
    covar.push_back(vector<vector<Vec3b> >());
    pr.push_back(vector<vector<Vec3b> >());

    for(int j=0;j<params.getCols();j++){
      mu[i].push_back(vector<Vec3b>());
      covar[i].push_back(vector<Vec3b>());
      pr[i].push_back(vector<Vec3b>());
    }
  }
}
int main(int argc, char** argv ){


  Mat src;                                                                      // Each frame.
  int framecount = 0;                                                           // For process end.
  vector<string> frames;                                                        // name of each frame.
  getframes("video/frames/",frames);
  int maxG = 0;                                                                 // maxG <= K, the max number of Gaussians.
  src = imread("video/frames/"+frames[framecount+2], CV_LOAD_IMAGE_COLOR);
  params.initParams(src.size().width,src.size().height);
  initialiseVec();

  while(true){
    if(framecount==10){
      cout<<"Done."<<endl;
      break;
    }
    src = imread("video/frames/"+frames[framecount+2], CV_LOAD_IMAGE_COLOR);

    // //To get intensity value at a point (x,y) // row => y, col => x.
    // // ordered BGR.
    // Vec3b intensity;
    // for(int y=0;y<100;y++){
    //   for(int x = 0;x<100;x++){
    //     intensity = src.at<Vec3b>(y,x);
    //     float blue = intensity.val[0];
    //     float green = intensity.val[1];
    //     float red = intensity.val[2];
    //     printf("blue: %f green: %f red: %f \n",blue,green,red );
    //   }
    // }
    framecount++;
  }
  imshow("",src);
  waitKey(0);
  return 0;
}
