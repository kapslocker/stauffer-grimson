#include <iostream>
#include <dirent.h>                 // for detecting frames
#include <vector>
#include <algorithm>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>

using namespace cv;
using namespace std;

const double PI = 4*atan(1);

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
    float learning_rate(){
        return alpha;
    }
    double init_covar(){                                                        // high initial covariance for a new distribution
        return 100;
    }
    double init_prior(){                                                        // low initial prior weight for a new distribution
        return alpha*0.1;
    }
}params;


//parameters of the gaussians, 3 for each pixel
vector<vector<vector<Vec3b> > > mu;
vector<vector<vector<double> > >covar;                                          // Mean and Covariance for each Gaussian at pixel (x,y).
vector<vector<vector<double> > > pr;                                       // Each Gaussian's contribution to the mixture at pixel (x,y)

void initialiseVec(){
     for(int i=0;i<params.getRows();i++){
        mu.push_back(vector<vector<Vec3b> >());
        covar.push_back(vector<vector<double> >());
        pr.push_back(vector<vector<double> >());

        for(int j=0;j<params.getCols();j++){
            mu[i].push_back(vector<Vec3b>());
            covar[i].push_back(vector<double>());
            pr[i].push_back(vector<double>());

            mu[i][j].resize(params.maxModes());
            covar[i][j].resize(params.maxModes());
            pr[i][j].resize(params.maxModes());
            for(int k = 0;k<params.maxModes();k++){
                mu[i][j][k] = Vec3b(0,0,0);
                covar[i][j][k] = params.init_covar();
                pr[i][j][k]    = params.init_prior();
            }
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


/*
* Evaluates the Gaussian, with covariance matrix as simply sig_squared * I .
*/
double eval_gaussian(Vec3b &X, Vec3b &u, double sig_squared){
    double s = 0;
    for(int i=0;i<3;i++){
        s += pow((X.val[i] - u.val[i]),2);
    }
    return exp(-0.5 * s/sig_squared )/sqrt(pow(2*PI,3)*sig_squared);
}

/*
*   w           -> a double array of size K, the array of weights for each Gaussian for each pixel.
*   X           -> value observed at time t.
*   u           -> double array of size K, the array containing means for each Gaussian for each pixel.
*   sig_squared -> double array of size K, containing the covariance for each Gaussian for each pixel.
*/
double eval_expectation(double *w, Vec3b &X, Vec3b *u, double *sig_squared){
    double sum = 0;
    for(int i=0;i<params.maxModes();i++){
        sum += w[i]*eval_gaussian(X, u[i], sig_squared[i]);
    }
    return sum;
}

/*
* Updates the value of mu, sig_squared for a specific distribution
* for the given pixel at time t
*/
//void update_gaussian(Vec3b X, Vec3b &u, double &sig_squared){
void update_gaussian(Vec3b X, int row, int col, int k){
    if(row == 50 and col == 50)
        cout<<"here"<<endl;
    Vec3b u = mu[row][col][k];

    double sig_squared = covar[row][col][k];

    if(sig_squared == 0){
        sig_squared = 100;
    }
    double p = params.learning_rate() * eval_gaussian(X, u, sig_squared);
    if(row==50 and col==50){
//        cout<<p<<endl;
    }
    u = (1-p)*u + p*X;
    sig_squared = (1-p)*sig_squared + pow(norm(X - u),2);

    mu[row][col][k] = u;
    covar[row][col][k] = sig_squared;
    pr[row][col][k] = (1-params.learning_rate())*pr[row][col][k] + params.learning_rate();

}

void replace_gaussian(Vec3b X, int row, int col, int k){
    if(row == 50 and col == 50)
        cout<<"Changed\n";
    mu[row][col][k] = X;
    covar[row][col][k] = params.init_covar();
    pr[row][col][k] = params.init_prior();
}
int main(int argc, char** argv ){


    VideoCapture capture = processVideo("video/umcp.mpg");

    keyboard = 0;

    double w = capture.get(CV_CAP_PROP_FRAME_WIDTH);
    double h = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
    params.initParams(w,h);
    initialiseVec();                                                            // create vectors for the gaussian params for each pixel

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
        for (int x = 0; x < params.getCols(); x++)
        {
            for (int y = 0; y < params.getRows(); y++)
            {
                Vec3b intensity = frame.at<Vec3b>(y,x);                         // i.e. X(t)

                if(y == 50 and x == 50){
//                    cout<<"intensity :"<<intensity<<endl;
                    cout<<"mean : "<<mu[y][x][0]<<"\t"<<mu[y][x][1]<<"\t"<<mu[y][x][2]<<endl;
                    cout<<"covar :"<<covar[y][x][0]<<"\t"<<covar[y][x][1]<<"\t"<<covar[y][x][2]<<endl;
                    cout<<"prior :"<<pr[y][x][0]<<"\t"<<pr[y][x][1]<<"\t"<<pr[y][x][2]<<endl;
                }
                /*  To read the color values use below code
                uchar blue = intensity.val[0];
                uchar green = intensity.val[1];
                uchar red = intensity.val[2];
                */
            // Find the distribution that matches the current value.
                int best_distr = -1, worst_distr = 0,maxsum = 0;
                for(int i=0;i<params.maxModes();i++){

                    double temp = 2.5 * sqrt(covar[y][x][i]);                // max allowed = 2.5 *sig.
                    Vec3b vec = intensity - mu[y][x][i];
                    bool found = true;
                    int sum = 0;
                    for(int j = 0;j<3;j++){                                     // deviation less than 2.5*sig for b,g,r
                        if( fabs(vec.val[j]) > temp ){
                            found = false;
                        }
                        sum += vec.val[j]*vec.val[j];
                    }
                    if(sum>maxsum){
                        worst_distr = i;
                        maxsum = sum;
                    }
                    if(found){
                        best_distr = i;
                        break;
                    }
                }
            // if distribution found :
                if(best_distr>-1)
                    update_gaussian(intensity,y,x,best_distr);
                else{
                    replace_gaussian(intensity,y,x,worst_distr);
                }
//                normalise_prior();
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

    return 0;
}
