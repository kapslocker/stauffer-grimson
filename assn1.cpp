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

using namespace cv;
using namespace std;

const double PI = 4*atan(1);

/*
* Contains parameters for the program
*/

bool Compare(const pair<double, double>&i, const pair<double,double>&j)
{
    return i.first > j.first;
}

Vec3b convert_3d_to_3b(Vec3d vec3d){
    Vec3b vec3b = Vec3b((uchar)vec3d.val[0],(uchar)vec3d.val[1],(uchar)vec3d.val[2]);
    return vec3b;
}
Vec3d convert_3b_to_3d(Vec3b vec3b){
    return Vec3d((double)vec3b.val[0],(double)vec3b.val[1],(double)vec3b.val[2]);
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
        K = 3;
        alpha = 0.1;
        T = 0.85;
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
        return Vec3d(0,0,0);
    }
    double init_covar(){                                                        // high initial covariance for a new distribution
        return 36.0;
    }
    double init_prior(){                                                        // low initial prior weight for a new distribution
        return 1.0/((double)maxModes());
    }
    float threshold(){
        return T;
    }
}params;

double norm_sq(Vec3d X){
    double s = 0;
    for(int i=0;i<3;i++){
        s += X.val[i]*X.val[i];
    }
    return s;
}

//parameters of the gaussians, 3 for each pixel
vector<vector<vector<Vec3d> > > mu;
vector<vector<vector<double> > >covar;                                          // Mean and Covariance for each Gaussian at pixel (x,y).
vector<vector<vector<double> > > pr;                                       // Each Gaussian's contribution to the mixture at pixel (x,y)


void initialiseVec(){
     for(int i=0;i<params.getRows();i++){
        mu.push_back(vector<vector<Vec3d> >());
        covar.push_back(vector<vector<double> >());
        pr.push_back(vector<vector<double> >());

        for(int j=0;j<params.getCols();j++){
            mu[i].push_back(vector<Vec3d>());
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
double eval_gaussian(Vec3d X, Vec3d u, double sig_squared){
    double s = norm_sq(X-u);
    double gaus = exp(s/(-2.0*sig_squared));
    gaus = gaus/sqrt(pow(2.0*PI,3.0)*sig_squared);
    return gaus;
}


/*
* Updates the value of mu, sig_squared for a specific distribution
* for the given pixel at time t
*/
void update_gaussian(Vec3d X, int row, int col, int k, bool is_match = false){

    Vec3d u = mu[row][col][k];
    double sig_squared = covar[row][col][k];
    double p = params.learning_rate() * eval_gaussian(X, u, sig_squared);

    u = (1.0-p)*u + p*X;
    sig_squared = (1.0-p)*sig_squared + p*norm_sq(X - u);
//    cout<<p*norm_sq(X-u)<<endl;
    mu[row][col][k] = u;
    covar[row][col][k] = sig_squared;
    //for the best match, w = (1-alpha)*w + alpha
    pr[row][col][k] = (1.0 -params.learning_rate())*pr[row][col][k] + ((double)is_match)*params.learning_rate();
}

void replace_gaussian(Vec3d X, int row, int col, int k){
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

// Global variables
Mat frame; //current frame
Mat bg_frame;
Mat fg_frame;


/*
* Perform the following at each pixel (y,x) y-> row, x -> col
*/
void perform_pixel(int y, int x){
    Vec3b value = frame.at<Vec3b>(y,x);                         // i.e. X(t)
    // Find the distribution that matches the current value.
    Vec3d intensity = convert_3b_to_3d(value);
    int worst_distr = 0,worst = 0;
    bool found = false;
    double min_dist_square = 1000000;
    for(int i=0;i<params.maxModes();i++){                           // Checking against each Gaussian
        Vec3d vec = intensity - mu[y][x][i];
        double dist = norm(vec,NORM_INF);
        double thres = 2.5 * covar[y][x][i];                // max allowed = 2.5 *sig.
        if( dist < thres ){
            found = true;
            update_gaussian(intensity,y,x,i,true);
        }
        else{
            update_gaussian(intensity,y,x,i);                               // do not add alpha to the other gaussians
        }
    }
    if(!found){
        // We need to replace a Gaussian now.
        Vec3d vec;
        double dist;
        for(int i=0;i<params.maxModes();i++){
            vec = intensity - mu[y][x][i];
            dist = norm(vec,NORM_INF);
            if(dist > worst){
                worst = dist;
                worst_distr = i;
            }
        }
        replace_gaussian(intensity,y,x,worst_distr);
        normalise_prior(y,x);
    }
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
    // modeling with the top match.
    bool f = false;
    double prior_sum = 0;

    Vec3d temp_3d = Vec3d(0,0,0);
    for(int j=0;j<B;j++){
//        if( norm( intensity - mu[y][x][gaus[j].second]) < 2.5*sqrt(covar[y][x][gaus[j].second]) ){
            temp_3d += pr[y][x][gaus[j].second]*mu[y][x][gaus[j].second];
            prior_sum += pr[y][x][gaus[j].second];
//            break;
//        }
    }
    temp_3d /= prior_sum;
    bg_frame.at<Vec3b>(y,x) = convert_3d_to_3b(temp_3d);
    fg_frame.at<Vec3b>(y,x) = value - bg_frame.at<Vec3b>(y,x);
    if(y == 200 and x == 200)
    cout<<B<<"\t"<<mu[y][x][0]<<"\t"<<mu[y][x][1]<<"\t"<<mu[y][x][2]<<"\t"<<pr[y][x][0]<<"\t"<<pr[y][x][1]<<"\t"<<pr[y][x][2]<<bg_frame.at<Vec3b>(y,x)<<endl;
}

int main(int argc, char** argv ){
    char keyboard; //input from keyboard

    VideoCapture capture = processVideo("video/umcp.mpg");

    keyboard = 0;
    double w = capture.get(CV_CAP_PROP_FRAME_WIDTH);
    double h = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
    params.initParams(w,h);
    // create vectors for the gaussian params for each pixel
    initialiseVec();
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
        //For each Pixel -- Perform update steps
        for (int x = 0; x < params.getCols(); x++ )
        {
            for (int y = 0; y < params.getRows(); y++ )
            {
                perform_pixel(y,x);
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

        imshow("Original", frame);
        imshow("Foreground", fg_frame);
        imshow("Background", bg_frame);

        //get the input from the keyboard
        keyboard = (char)waitKey( 30 );
    }


    //delete capture object
    capture.release();

    return 0;
}
