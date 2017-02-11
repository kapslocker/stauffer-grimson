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

bool Compare(const pair<double, double>&i, const pair<double,double>&j)
{
    return i.first > j.first;
}

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
        alpha = 0.1f;
        T = 0.95f;
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
    Vec3b init_mean(){
        return Vec3b(0,0,0);
    }
    double init_covar(){                                                        // high initial covariance for a new distribution
        return 360.0f;
    }
    double init_prior(){                                                        // low initial prior weight for a new distribution
        return 1/maxModes();
    }
    float threshold(){
        return T;
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
                mu[i][j][k] = params.init_mean();
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



/*
* Evaluates the Gaussian, with covariance matrix as simply sig_squared * I .
*/
double eval_gaussian(Vec3b X, Vec3b u, double sig_squared){
    double s = pow(norm(X - u),2);
    return exp(-0.5 * s/sig_squared )/sqrt(pow(2*PI,3)*sig_squared);
}


/*
* Updates the value of mu, sig_squared for a specific distribution
* for the given pixel at time t
*/
//void update_gaussian(Vec3b X, Vec3b &u, double &sig_squared){
void update_gaussian(Vec3b X, int row, int col, int k){
    Vec3b u = mu[row][col][k];

    double sig_squared = covar[row][col][k];

    if(sig_squared == 0){
        sig_squared = 100;
    }
    double p = params.learning_rate() * eval_gaussian(X, u, sig_squared);
    u = (1-p)*u + p*X;
    sig_squared = (1-p)*sig_squared + pow(norm(X - u),2);

    mu[row][col][k] = u;
    covar[row][col][k] = sig_squared;
    pr[row][col][k] = (1-params.learning_rate())*pr[row][col][k] + params.learning_rate();

}

void replace_gaussian(Vec3b X, int row, int col, int k){
    mu[row][col][k] = X;
    covar[row][col][k] = params.init_covar();
    pr[row][col][k] = params.init_prior();

}

/*
* Normalises prior probabilities of the Gaussian mixture at pixel (row,col).
*/
void normalise_prior(int row, int col){
    double s = 0.0;
    for(int i=0;i<params.maxModes();i++){
        s += pr[row][col][i];
    }
    if(s != 0 )
        for(int i=0;i<params.maxModes();i++)
            pr[row][col][i] /= s;
}

int main(int argc, char** argv ){

    // Global variables
    Mat frame; //current frame
    Mat bg_frame;
    Mat fg_frame;

    char keyboard; //input from keyboard

    VideoCapture capture = processVideo("video/umcp.mpg");

    keyboard = 0;

    double w = capture.get(CV_CAP_PROP_FRAME_WIDTH);
    double h = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
    params.initParams(w,h);
    initialiseVec();                                                            // create vectors for the gaussian params for each pixel

    bg_frame = Mat(params.getRows(), params.getCols(), CV_8UC3);
    fg_frame = Mat(params.getRows(), params.getCols(), CV_8UC3);
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
        for (int x = 0; x < params.getCols(); x++ )
        {
            for (int y = 0; y < params.getRows(); y++ )
            {
                Vec3b intensity = frame.at<Vec3b>(y,x);                         // i.e. X(t)
/*
                if(y == 100 and x == 100){
                    cout<<"intensity :"<<intensity<<endl;
                    cout<<"mean : "<<mu[y][x][0]<<"\t"<<mu[y][x][1]<<"\t"<<mu[y][x][2]<<endl;
                    cout<<"covar :"<<covar[y][x][0]<<"\t"<<covar[y][x][1]<<"\t"<<covar[y][x][2]<<endl;
                    cout<<"prior :"<<pr[y][x][0]<<"\t"<<pr[y][x][1]<<"\t"<<pr[y][x][2]<<endl;
                }
*/            // Find the distribution that matches the current value.
                int best_distr = -1, worst_distr = 0,worst_dist_sq = 0;

                for(int i=0;i<params.maxModes();i++){
                    Vec3b vec = intensity - mu[y][x][i];
                    double dist_sq = pow(norm(vec),2);
                    double thres_sq = 6.25 * covar[y][x][i];                // max allowed = 2.5 *sig.
                    if( dist_sq < thres_sq ){
                        update_gaussian(intensity,y,x,i);
                        break;
//                        best_distr = i;
                    }
                    if(dist_sq > worst_dist_sq){
                        worst_distr = i;
                        worst_dist_sq = dist_sq;
                    }
                }
            // if distribution found :
                if(best_distr>-1){
//                    update_gaussian(intensity,y,x,best_distr);
                }
                else{
                    replace_gaussian(intensity,y,x,worst_distr);
                //  Normalise only if a Gaussian is replaced!
                    normalise_prior(y,x);
                }
//                if(y == 100 and x == 100)
//                    circle( frame,Point(50,50),10.0,Scalar( 0, 0, 255 ));

                vector<pair<double,double> > gaus(params.maxModes());
                for(int i=0;i<params.maxModes();i++){
                    gaus[i] = make_pair( pr[y][x][i]/sqrt(covar[y][x][i]) , i);  // sort Gaussians wrt w/sigma, identifier used
                                                                                // is the index of that Gaussian
                }
                sort(gaus.begin(), gaus.end(), Compare);

                float sum = 0;
                int B;
                for(B=0;B<params.maxModes();B++){
                    sum += gaus[B].first;
                    if(sum > params.threshold())    break;
                }
                bool f = false;
                for(int j=0;j<B;j++){
                    if( norm( intensity - mu[y][x][gaus[j].second]) < 2.5*sqrt(covar[y][x][gaus[j].second]) ){
                        bg_frame.at<Vec3b>(y,x) = mu[y][x][gaus[j].second];
                        f = true;
                        break;
                    }
                }
                fg_frame.at<Vec3b>(y,x) = intensity - bg_frame.at<Vec3b>(y,x);
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

        imshow("Foreground", fg_frame);
        imshow("Background", bg_frame);

        //get the input from the keyboard
        keyboard = (char)waitKey( 30 );
    }


    //delete capture object
    capture.release();

    return 0;
}
