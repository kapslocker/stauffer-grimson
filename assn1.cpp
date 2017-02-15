#include <iostream>
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
#include <omp.h>

using namespace cv;
using namespace std;

const double PI = 4*atan(1);

/*
* Contains parameters for the program
*/

bool Compare(const pair<Vec3f, double>&i, const pair<Vec3f,double>&j)
{
    return i.second > j.second;
}

class Params{
    int width;
    int height;
    int K;
    double alpha, T;
    public:
        void initParams(int w, int h){
        width = w;
        height = h;
        K = 5;
        alpha = 0.01;
        T = 0.5;
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
    double learning_rate(){
        return alpha;
    }
    Vec3d init_mean(){
        return Vec3d(0.0,0.0,0.0);
    }
    double init_covar(){                                                        // high initial covariance for a new distribution
        return 10.0;
    }
    double init_prior(){                                                        // low initial prior weight for a new distribution
        return 1.0/((double)maxModes());
    }
    double threshold(){
        return T;
    }
}params;


//parameters of the gaussians, 3 for each pixel
Vec3d ***mu;                                                                 // height * width * modes * 3
double ***covar;                                                                // Mean and Covariance for each Gaussian at pixel (x,y).
double *** pr;                                                                  // Each Gaussian's contribution to the mixture at pixel (x,y)


void initialiseVec(){
    mu = new Vec3d **[params.getRows()];
    covar = new double **[params.getRows()];
    pr = new double **[params.getRows()];

     for(int i=0;i<params.getRows();i++){
        mu[i] = new Vec3d *[params.getCols()];
        covar[i] = new double *[params.getCols()];
        pr[i]   = new double *[params.getCols()];

        for(int j=0;j<params.getCols();j++){

            mu[i][j] = new Vec3d[params.maxModes()];
            covar[i][j] = new double[params.maxModes()];
            pr[i][j] = new double[params.maxModes()];

            for(int k = 0;k<params.maxModes();k++){
                mu[i][j][k] = params.init_mean();
                covar[i][j][k] = params.init_covar();
                pr[i][j][k] = params.init_prior();
            }
        }
    }
}

VideoCapture processVideo(string fileName){
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
double eval_gaussian(Vec3d X,int row,int col, int k){
    double sig_square = covar[row][col][k];
    double gaus  = exp(-0.5 * norm(X - mu[row][col][k] , NORM_L2SQR) / sig_square );
    gaus /= pow(2*PI*sig_square,1.5);
    return gaus;
}


/*
* Updates the value of mu, sig_squared for a specific distribution
* for the given pixel at time t
*/
void update_gaussian(Vec3d X, int row, int col, int k){

    double rho = params.learning_rate() * eval_gaussian(X, row, col, k);

    mu[row][col][k] = (1.0 - rho) * mu[row][col][k] + rho * X;

    covar[row][col][k] = (1.0 - rho) * covar[row][col][k] + rho * norm(X - mu[row][col][k] , NORM_L2SQR);
}

void replace_gaussian(Vec3d X, int row, int col, int k){
    mu[row][col][k] = X;
    covar[row][col][k] = params.init_covar()*10;
    pr[row][col][k] = params.init_prior()/10;
}

/*
* Normalises prior probabilities of the Gaussian mixture at pixel (row,col).
*/
void normalise_prior(int row, int col){
    double s = 0.0;
    for(int i = 0; i < params.maxModes(); i++){
        s += pr[row][col][i];
    }
    if(s == 0){
        cout<<"Trouble at: "<< row <<" "<< col <<endl;
        return ;
    }
    for(int i = 0; i < params.maxModes(); i++)
        pr[row][col][i] /= s;
}

// Global variables
Mat frame; //current fram
Mat bg_frame;
Mat fg_frame;


void update_prior(int y, int x, int match){
    for(int i = 0; i < params.maxModes(); i++){
        pr[y][x][i] = (1.0 - params.learning_rate()) * pr[y][x][i];
        if(i == match)
            pr[y][x][i] += params.learning_rate();
    }
    normalise_prior(y,x);
}


bool compare_pr(const pair<double,double> &i, const pair<double,double> &j){
    return i.second > j.second;
}

void sort_model(int y, int x){
        vector<pair<Vec3f,double> > mu_values(params.maxModes());
        vector<pair<double,double> > pr_values(params.maxModes());
        vector<pair<double,double> > covar_values(params.maxModes());

        for(int i=0;i<params.maxModes();i++){
            mu_values[i] = make_pair(mu[y][x][i], pr[y][x][i] / sqrt(covar[y][x][i]));
            pr_values[i] = make_pair(pr[y][x][i], pr[y][x][i] / sqrt(covar[y][x][i]));
            covar_values[i] = make_pair(covar[y][x][i], pr[y][x][i] / sqrt(covar[y][x][i]));
        }
        sort(mu_values.begin(), mu_values.end(), Compare);
        sort(pr_values.begin(), pr_values.end(), compare_pr);
        sort(covar_values.begin(),covar_values.end(), compare_pr);

        for(int i=0; i < params.maxModes(); i++){
            mu[y][x][i] = mu_values[i].first;
            pr[y][x][i] = pr_values[i].first;
            covar[y][x][i] = covar_values[i].first;
        }

}

void create_model(int y, int x, int found){
    // find B = argmin_i s.t. sum_i pr(i) > T
    double sum = 0;
    Vec3d new_3d = Vec3d(0.0,0.0,0.0);
    for( int i=0; i < params.maxModes(); i++){
        sum += pr[y][x][i];
        new_3d += pr[y][x][i] * mu[y][x][i];
        if(sum > params.threshold()){
            break;
        }
    }

    bg_frame.at<Vec3b>(y,x) = Vec3b(new_3d/sum);
    if(found){
        fg_frame.at<Vec3b>(y,x) = Vec3b(0,0,0);
    }
    else{
        fg_frame.at<Vec3b>(y,x) = Vec3b(255,255,255);
    }
}

bool check_match(Vec3d X, int row, int col, int k){

    double ns1 = norm(X - mu[row][col][k], NORM_INF);
    double sig = 2.5*sqrt(covar[row][col][k]);
    if(ns1 < sig){                                     // L- inf norm. <  2.5 * sqrt(covar)
        return true;
    }
    return false;
}

/*
* Perform the following at each pixel (y,x) y-> row, x -> col
*/
void perform_pixel(int y,int x){

    Vec3b value = frame.at<Vec3b>(y,x);

    Vec3d X = Vec3d(value);
    bool found = false;
    int match = -1;
    for(int i=0;i<params.maxModes();i++){
        if(check_match(X,y,x,i)){
            found = true;
            match = i;
            // update gaussian to which the pixel matches.
            update_prior(y,x,match);
            update_gaussian(X, y,x,match);
            break;
        }
    }
    if(found){
        // no need to sort if not found a match. No gaussians shall be disturbed.
        sort_model(y,x);
    }
    if(!found){

        // replace the gaussian with min significant (significant = pr/sqrt(covar)) -> replace the last gaussian.
        match = params.maxModes() - 1;
        replace_gaussian(X,y,x,match);
        update_prior(y,x,match);
    }


    create_model(y,x,found);

}

int main(int argc, char** argv ){

	int numthreads = omp_get_max_threads();

    char keyboard; //input from keyboard
    string vid_loc = "video/test4.avi";
    VideoCapture capture;
    capture.release();
    capture = processVideo(vid_loc);
    keyboard = 0;
    double w = capture.get(CV_CAP_PROP_FRAME_WIDTH);
    double h = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
    params.initParams(w,h);
    // create vectors for the gaussian params for each pixel
    initialiseVec();
    bg_frame = Mat(params.getRows(), params.getCols(), CV_8UC3);
    fg_frame = Mat(params.getRows(), params.getCols(), CV_8UC3);
    // the main while loop inside which the video processing happens
    while( keyboard != 'q' and keyboard != 27 ){
        //read the current frame
        if(!capture.read(frame)) {
            cerr << "Unable to read next frame." << endl;
            cerr << "Exiting..." << endl;
            exit(EXIT_FAILURE);
        }
        //PROCESS HERE
        //For each Pixel -- Perform update steps
        #pragma omp parallel for num_threads(numthreads)
        for (int x = 0; x < params.getCols(); x++ )
        {
        	#pragma omp parallel for num_threads(numthreads)
            for (int y = 0; y < params.getRows(); y++ )
            {
                perform_pixel(y,x);
            }
        }
        fg_frame = bg_frame - frame;
        threshold(fg_frame, fg_frame, 20, 255, THRESH_BINARY );
        Mat grayfg = Mat(params.getRows(), params.getCols(), CV_8U);
        cvtColor( fg_frame, grayfg, CV_BGR2GRAY );
        threshold(grayfg, grayfg, 250, 255, THRESH_BINARY );

        medianBlur(grayfg,grayfg,5);
        imshow("Original", frame);
        imshow("Foreground", grayfg);
        imshow("Background", bg_frame);

        //get the input from the keyboard
        keyboard = (char)waitKey(1);
    }


    //delete capture object
    capture.release();

    return 0;
}
