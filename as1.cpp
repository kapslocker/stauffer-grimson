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

// Global variables
Mat frame; //current frame
Mat bg_frame;
Mat fg_frame;


bool Compare(const pair<double, double>&i, const pair<double,double>&j)
{
    return i.first > j.first;
}


//get the 2 norm
double norm_sq(Vec3b X){
    double s = 0;
    s = norm(X);
    s *= s;
    // for(int i=0;i<3;i++){
    //     s += X.val[i]*X.val[i];
    // }
    return s;
}


/*
* Evaluates the Gaussian, with covariance matrix as simply sig_squared * I .
*/
double eval_gaussian(Vec3b &X, Vec3b &u, double sig_squared){
    double s = norm_sq(X-u);
    double gaus = exp(-0.5 * s / sig_squared);
    gaus = gaus/pow(2.0 * PI * sig_squared, 1.5);
    return gaus;
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
        alpha = 0.05;
        T = 0.3;
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
    Vec3b init_mean(){
        return Vec3b(0,0,0);
    }
    double init_covar(){                                                        // high initial covariance for a new distribution
        return 400.0;
    }
    double init_prior(){                                                        // low initial prior weight for a new distribution
        return 1.0/((double)maxModes());
    }
    double threshold(){
        return T;
    }
}params;



//parameters of the gaussians, 3 for each pixel
Vec3b ***mu;                          // height * width * modes * 3
double ***covar;                      // Mean and Covariance for each Gaussian at pixel (x,y).
double *** pr;                        // Each Gaussian's contribution to the mixture at pixel (x,y)



void initialiseVec(){
    mu = new Vec3b **[params.getRows()];
    covar = new double **[params.getRows()];
    pr = new double **[params.getRows()];

     for(int i=0;i<params.getRows();i++){
        mu[i] = new Vec3b *[params.getCols()];
        covar[i] = new double *[params.getCols()];
        pr[i]   = new double *[params.getCols()];

        for(int j=0;j<params.getCols();j++){

            mu[i][j] = new Vec3b [params.maxModes()];
            covar[i][j] = new double [params.maxModes()];
            pr[i][j] = new double [params.maxModes()];

            for(int k = 0;k<params.maxModes();k++){

                mu[i][j][k] = params.init_mean();
                covar[i][j][k] = params.init_covar();
                pr[i][j][k] = params.init_prior();
            }
        }
    }
}

void update_gaussian(Vec3b intensity, int y, int x, int k, bool match){
	//update responsibility
	pr[y][x][k] = (1 - params.learning_rate())*pr[y][x][k] + params.learning_rate()*match;


	if(match){
		//update mean and sigma
		//parameter p
		double p = params.learning_rate() * eval_gaussian(intensity, mu[y][x][k], covar[y][x][k]);

		mu[y][x][k]	=(1.0 - p)* (Vec3d) mu[y][x][k] + p* (Vec3d) intensity;

		covar[y][x][k] = (1.0 - p)*covar[y][x][k] + p*norm_sq(intensity - mu[y][x][k]);
	}
	else{
		for(int i =0; i< params.maxModes() ; i++){
			if(i!=k){
				pr[y][x][k] = (1 - params.learning_rate())*pr[y][x][k];
			}
		}
	}
}



void replace_gaussian(Vec3b X, int row, int col, int k){
    mu[row][col][k] = X;
    covar[row][col][k] = params.init_covar();
    pr[row][col][k] = params.init_prior()/10;
}

void normalize_weights(int row, int col){
    double s = 0.0;
    for(int i=0;i<params.maxModes();i++){
        s += pr[row][col][i];
    }
    if(s != 0 )
        for(int i=0;i<params.maxModes();i++)
            pr[row][col][i] /= s;
}




void perform_pixel(int y, int x){
	Vec3b intensity = frame.at<Vec3b>(y,x);

	bool match = false;


	for(int i =0; i< params.maxModes() ; i++){

		double dist = norm(intensity - mu[y][x][i]);

		double thresh = 2.5 * sqrt(covar[y][x][i]);

		if(dist < thresh){
			match = true;

			update_gaussian(intensity, y,x,i,match);
			normalize_weights(y,x);

			break;
		}
	}

	if(!match){
		Vec3d vec;
        double dist;
        double worst = 0;
        int worst_distr = -1;
        for(int i=0;i<params.maxModes();i++){
            vec = (intensity - mu[y][x][i]);
            dist = norm(vec);
            if(dist > worst){
                worst = dist;
                worst_distr = i;
            }
        }
        replace_gaussian(intensity,y,x,worst_distr);
        normalize_weights(y,x);
	}



	// to get the B best gaussians based on threshold
	vector<pair<double,double> > gaus(params.maxModes());

    for(int i=0;i<params.maxModes();i++){
        gaus[i] = make_pair( pr[y][x][i]/sqrt(covar[y][x][i]) , i);             // sort Gaussians wrt w/sigma, identifier used
                                                                                // is the index of that Gaussian
    }
    sort(gaus.begin(), gaus.end(), Compare);

    float sum = 0;
    int B;
    for(B=0;B<params.maxModes();B++){
        sum += gaus[B].first * sqrt(covar[y][x][(int)gaus[B].second]);
        if(sum > params.threshold())    break;
    }

    double prior_sum = 0.0;

    Vec3b temp_3d = Vec3b(0,0,0);
    int pos;    
    for(int j=0;j<=B;j++){
        pos = gaus[j].second;
        temp_3d += pr[y][x][pos]* (Vec3d) mu[y][x][pos];
        prior_sum += pr[y][x][pos];
    }
    temp_3d /= prior_sum;
    
    

    bg_frame.at<Vec3b>(y,x) = (temp_3d);
    // if(!match){
    //     fg_frame.at<Vec3b>(y,x) = Vec3b(255,255,255);
    // }else{
    //     fg_frame.at<Vec3b>(y,x) = Vec3b(0,0,0);
    // }

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

Ptr<BackgroundSubtractor> pMOG2; //MOG2 Background subtractor

int main(int argc, char** argv ){
    char keyboard; //input from keyboard
 //   string vid_loc = "video/umcp.mpg";
    string vid_loc = "video/pets01.wmv";

    VideoCapture capture = processVideo(vid_loc);
    keyboard = 0;
    double w = capture.get(CV_CAP_PROP_FRAME_WIDTH);
    double h = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
    params.initParams(w,h);
    // create vectors for the gaussian params for each pixel
    initialiseVec();
    bg_frame = Mat(params.getRows(), params.getCols(), CV_8UC3);
//    fg_frame = Mat(params.getRows(), params.getCols(), CV_8UC3);

	pMOG2 = createBackgroundSubtractorMOG2(); //MOG2 approach
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
        pMOG2->apply(frame, fg_frame);
        //The output area -- output video and whatever else we want
        //get the frame number and write it on the current frame
        imshow("Original", frame);
        imshow("Foreground", fg_frame);
        imshow("Background", bg_frame);

        //get the input from the keyboard
        keyboard = (char)waitKey( 1 );
    }


    //delete capture object
    capture.release();

    return 0;
}
