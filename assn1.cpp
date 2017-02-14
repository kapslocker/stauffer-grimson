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

const float PI = 4*atan(1);

/*
* Contains parameters for the program
*/

bool Compare(const pair<float, float>&i, const pair<float,float>&j)
{
    return i.first > j.first;
}

Vec3f dabs(Vec3f X){
    return Vec3f(fabs(X.val[0]),fabs(X.val[1]),fabs(X.val[2]));
}
// Vec3b convert_3f_to_3b(Vec3f vec3f){
//     Vec3b vec3b = Vec3b((uchar)vec3f.val[0],(uchar)vec3f.val[1],(uchar)vec3f.val[2]);
//     return vec3b;
// }
// Vec3f convert_3b_to_3f(Vec3b vec3b){
//     return Vec3f((float)vec3b.val[0],(float)vec3b.val[1],(float)vec3b.val[2]);
// }
class Params{
    int width;
    int height;
    int K;
    float alpha, T;
    public:
        void initParams(int w, int h){
        width = w;
        height = h;
        K = 4;
        alpha = 0.015f;
        T = 0.85f;
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
    float init_mean(){
        return 0.0;
    }
    float init_covar(){                                                        // high initial covariance for a new distribution
        return 100.0;
    }
    float init_prior(){                                                        // low initial prior weight for a new distribution
        return 1.0/((double)maxModes());
    }
    float threshold(){
        return T;
    }
}params;


//parameters of the gaussians, 3 for each pixel
float ****mu;                                                                 // height * width * modes * 3
float ***covar;                                                                // Mean and Covariance for each Gaussian at pixel (x,y).
float *** pr;                                                                  // Each Gaussian's contribution to the mixture at pixel (x,y)


void initialiseVec(){
    mu = new float ***[params.getRows()];
    covar = new float **[params.getRows()];
    pr = new float **[params.getRows()];

     for(int i=0;i<params.getRows();i++){
        mu[i] = new float **[params.getCols()];
        covar[i] = new float *[params.getCols()];
        pr[i]   = new float *[params.getCols()];

        for(int j=0;j<params.getCols();j++){

            mu[i][j] = new float *[params.maxModes()];
            covar[i][j] = new float [params.maxModes()];
            pr[i][j] = new float [params.maxModes()];

            for(int k = 0;k<params.maxModes();k++){

                mu[i][j][k] = new float[3];
                for(int l=0;l<3;l++){
                    mu[i][j][k][l] = params.init_mean();
                }
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

float norm_sq(float r, float g, float b){
    float s = 0;
    s = r*r + g*g + b*b;
    return s;
}


/*
* Evaluates the Gaussian, with covariance matrix as simply sig_squared * I .
*/
float eval_gaussian(float col_arr[],int row,int col, int gaus){
    float arr[3] ;
    for(int i=0;i<3;i++){
        arr[i] = mu[row][col][gaus][i];
    }
    float sig_square = covar[row][col][gaus];
    gaus  = exp(-0.5 * norm_sq(col_arr[0] - arr[0], col_arr[1] - arr[1], col_arr[2] - arr[2]) / sig_square );
    gaus /= pow(2*PI*sig_square,1.5);
    return gaus;
}


/*
* Updates the value of mu, sig_squared for a specific distribution
* for the given pixel at time t
*/
void update_gaussians(float arr[], int row, int col, int k){
    float rho = params.learning_rate() * eval_gaussian(arr, row, col, k);

    mu[row][col][k][0] = (1- rho) * mu[row][col][k][0] + rho * arr[0];
    mu[row][col][k][1] = (1- rho) * mu[row][col][k][1] + rho * arr[1];
    mu[row][col][k][2] = (1- rho) * mu[row][col][k][2] + rho * arr[2];

    float r = arr[0] - mu[row][col][k][0],g = arr[1] - mu[row][col][k][1], b = arr[2] - mu[row][col][k][2];
    covar[row][col][k] = (1- rho) * covar[row][col][k] + rho * norm_sq(r,g,b);

}

void replace_gaussian(float X[3], int row, int col, int k){
    mu[row][col][k][0] = X[0];
    mu[row][col][k][1] = X[1];
    mu[row][col][k][2] = X[2];

    covar[row][col][k] = params.init_covar();
    pr[row][col][k] = params.init_prior();
}

/*
* Normalises prior probabilities of the Gaussian mixture at pixel (row,col).
*/
void normalise_prior(int row, int col){
    float s = 0.0;
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


void update_prior(int y,int x,int match){
    for(int i=0;i<params.maxModes();i++){
        pr[y][x][i] = (1-params.learning_rate())*pr[y][x][i];
        if(i == match)
            pr[y][x][i] += params.learning_rate();
    }
    normalise_prior(y,x);
}

bool check_match(float r, float g, float b, int row, int col, int gaus){

    float mu_red = mu[row][col][gaus][0];
    float mu_green = mu[row][col][gaus][1];
    float mu_blue = mu[row][col][gaus][2];


    float ns = norm_sq(r- mu_red, g - mu_green, b - mu_blue);

    if(ns<6.25*covar[row][col][gaus]){
        return true;
    }
    return false;
}

void create_model(int y,int x,int found){

    vector<pair<float,float> > values(params.maxModes());
    for(int i=0;i<params.maxModes();i++){
        values[i] = make_pair(pr[y][x][i] / sqrt(covar[y][x][i]) , i);
    }
    sort(values.begin(),values.end(),Compare);

    float sum = 0;
    int B = 0;
    for(B=0;B<params.maxModes();B++){
        sum += pr[y][x][(int)values[B].second];
        if(sum > params.threshold()){
            break;
        }
    }
    float new_3f[3] = {0.0,0.0,0.0};
    for(int i=0;i<=B;i++){
        for(int j=0;j<3;j++)
            new_3f[j] += pr[y][x][(int)values[i].second]*mu[y][x][(int)values[i].second][j];
    }
    bg_frame.at<Vec3b>(y,x) = Vec3b((int)new_3f[0],(int)new_3f[1],(int)new_3f[2]);
    if(found){
        fg_frame.at<Vec3b>(y,x) = Vec3b(0,0,0);
    }
    else{
        fg_frame.at<Vec3b>(y,x) = Vec3b(255,255,255);
    }


}


/*
* Perform the following at each pixel (y,x) y-> row, x -> col
*/
void perform_pixel_new(int y,int x){

    Vec3b value = frame.at<Vec3b>(y,x);
    float r = (float)value.val[0];
    float g = (float)value.val[1];
    float b = (float)value.val[2];
    float colours[3] = {r,g,b};
    bool found = false;
    int match = -1;
    int least_probable = -1;
    for(int i=0;i<params.maxModes();i++){
        if(check_match(r,g,b,y,x,i)){
            found = true;
            match = i;
            break;
        }
    }
    if(!found){
        float worst = 0;
        for(int i=0;i<params.maxModes();i++){
            float dist = norm_sq(r - mu[y][x][i][0],g - mu[y][x][i][1],b - mu[y][x][i][2]);
            if( dist > worst){
                worst = dist;
                least_probable = i;
            }
        }
        replace_gaussian(colours,y,x,least_probable);
    }

    update_prior(y,x,match);


    if(match > -1){
        update_gaussians(colours, y,x,match);
    }

    create_model(y,x,found);

}

int main(int argc, char** argv ){
    char keyboard; //input from keyboard
    string vid_loc = "video/umcp.mpg";
    VideoCapture capture = processVideo(vid_loc);
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
                perform_pixel_new(y,x);
            }
        }

        //The output area -- output video and whatever else we want
        //get the frame number and write it on the current frame
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
