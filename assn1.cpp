#include <iostream>
#include <dirent.h>                 // for detecting frames
#include <vector>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>

using namespace cv;
using namespace std;



/**
*  @params dir, frames
*  Fill vector frames with file names of images from directory dir.
*/
/*
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
*/

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


//paramaters of the gaussians, 3 for each pixel
vector<vector<vector<Vec3b> > > mu,covar;                  // Mean and Covariance for each Gaussian at pixel (x,y).
vector<vector<vector<Vec3b> > > pr;                        // Each Gaussian's contribution to the mixture at pixel (x,y)

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

VideoCapture processVideo(char* fileName){
    VideoCapture capture(fileName);
        if(!capture.isOpened()){
            //error in opening the video input
            cerr << "Unable to open video file: " << fileName << endl;
            exit(EXIT_FAILURE);
        }   

    return capture;
}

// Global variables
Mat frame; //current frame

char keyboard; //input from keyboard


int main(int argc, char** argv ){

    VideoCapture capture = processVideo("video/umcp.mpg");

    keyboard = 0;

    initialiseVec();
    params.initParams(frame.size().width,frame.size().height);

    // the main while loop inside which the video processing happens
    while( keyboard != 'q' && keyboard != 27 ){
        //read the current frame
        if(!capture.read(frame)) {
            cerr << "Unable to read next frame." << endl;
            cerr << "Exiting..." << endl;
            exit(EXIT_FAILURE);
        }

        //PROCESS HERE

        //For each Pixel -- Do the EM steps
        for (int x = 0; x < params.getCols(); ++x)
        {
            for (int y = 0; y < params.getRows(); ++y)
            {
                Vec3b intensity = frame.at<Vec3b>(y,x);
                
                /*  To read the color values use below code
                uchar blue = intensity.val[0];
                uchar green = intensity.val[1];
                uchar red = intensity.val[2];
                */

            //Expectation step -- maximize pr based off values of pixel

            //Maximization step -- maximize mu, pi, and covar based off the pr -- use formulas


            }
        }
        

        


        //The output area -- output video and whatever else we want
        //get the frame number and write it on the current frame
        stringstream ss;
        rectangle(frame, cv::Point(10, 2), cv::Point(100,20),
                  cv::Scalar(255,255,255), -1);
        ss << capture.get(CAP_PROP_POS_FRAMES);
        string frameNumberString = ss.str();
        putText(frame, frameNumberString.c_str(), cv::Point(15, 15),
                FONT_HERSHEY_SIMPLEX, 0.5 , cv::Scalar(0,0,0));
        //show the current frame and the fg masks
        imshow("Frame", frame);

        //get the input from the keyboard
        keyboard = (char)waitKey( 30 );
    }


    //delete capture object
    capture.release();


/*
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

  */
  return 0;
}
