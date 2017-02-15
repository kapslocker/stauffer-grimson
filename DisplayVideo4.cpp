#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <math.h>
#include <pthread.h>

using namespace cv;
using namespace std;

int k = 4;
float alpha = 0.05;
float weightThresh = 0.5;

float ***muBlue;
float ***muGreen;
float ***muRed;
float ***w;
float ***sigma;

Mat frame;
Mat detect;
Mat bg;
Mat fore;

double dHeight;
double dWidth;

string video;

VideoCapture cap;


void initializeVars();
bool setVideo();
bool algorithm();
void updateBackgroundForegroundModel(int y, int x);
bool match(int xVal, float mu, float sigma);
void updateWeights(int x, int y, int M);
void sortByWeights(int x, int y);
void updateMuSigma(int x, int y, int i, float rho, int blue, int green, int red);
bool fitGaussian(int x, int y, Vec3b pix);


int main(int argc, char* argv[])
{

	if(argc <= 1) {
		cout << "Enter video name in command line" << endl;
		return -1;
	}

	video = argv[1];
	
	if (!setVideo()) {
		cout << "Cannot open the video" << endl;
		return -1;
	}
	
	initializeVars();

	if(!algorithm()) {
		return -1;
	} else {
		return 0;
	}

}

void initializeVars() {

	dWidth = cap.get(CV_CAP_PROP_FRAME_WIDTH); //get the width of frames of the video
	dHeight = cap.get(CV_CAP_PROP_FRAME_HEIGHT); //get the height of frames of the video
	
	muBlue = new float **[(int)dWidth];
	muGreen = new float **[(int)dWidth];
	muRed = new float **[(int)dWidth];
	w = new float **[(int)dWidth];
	sigma = new float **[(int)dWidth];
	
	for(int i = 0 ; i < (int)dWidth ; ++i) {
		muBlue[i] = new float *[(int)dHeight];
		muGreen[i] = new float *[(int)dHeight];
		muRed[i] = new float *[(int)dHeight];
		w[i] = new float *[(int)dHeight];
		sigma[i] = new float *[(int)dHeight];
	
		for(int j = 0 ; j<(int)dHeight ; ++j) {
			muBlue[i][j] = new float [k];
			muGreen[i][j] = new float [k];
			muRed[i][j] = new float [k];
			w[i][j] = new float [k];
			sigma[i][j] = new float [k];
			
			for(int l = 0 ; l < k ; ++l) {
				muBlue[i][j][l] = 0.0;
				muGreen[i][j][l] = 0.0;
				muRed[i][j][l] = 0.0;
				w[i][j][l] = (float)(1.0/(float)k);
				sigma[i][j][l] = 6.0;
			}
		}
	}

	detect = Mat((int)dHeight, (int)dWidth, CV_8UC1, Scalar(0));
	bg = Mat((int)dHeight, (int)dWidth, CV_8UC3, Scalar(255, 255, 255));
	fore = Mat((int)dHeight, (int)dWidth, CV_8UC1, Scalar(0));
}

bool setVideo() {
	cap.release();
	cap = VideoCapture(video);

	if (!cap.isOpened()) {
		return false;
	}

	return true;
}

bool algorithm() {
	while (1) {

		bool bSuccess = cap.read(frame); // read a new frame from video

		if (!bSuccess) //if not success, break loop
		{
			cout << "Video stream terminated. Resetting video." << endl;
			if(setVideo()) {
				continue;
			} else {
				return false;
			}
		}

		imshow("Video", frame); //show the frame in "MyVideo" window

		for(int x = 0 ; x < (int)dWidth ; ++x) {
			for(int y = 0 ; y < (int)dHeight ; ++y) {
				Vec3b pix = frame.at<Vec3b>(y,x);

				if(fitGaussian(x,y,pix)) {
					// create a BLACK pixel
					detect.at<uchar>(y,x) = 0;
				} else {
					// create a WHITE pixel
					detect.at<uchar>(y,x) = 255;
				}
				
				updateBackgroundForegroundModel(y,x);
			}
		}

//		medianBlur(fore, fore, 5);
		imshow("BG model", bg);
		imshow("foreground", fore);

		if (waitKey(1) == 27) // If 'esc' key is pressed, break loop
		{
			cout << "esc key is pressed by user" << endl;
			return false; 
		}
	}
	return true;
}

void updateWeights(int x, int y, int M) {
	float sum = 0;
	for(int i = 0 ; i < k ; ++i) {
		if(i == M) {
			w[x][y][i] = (1.0-alpha)*w[x][y][i] + alpha;
		} else {
			w[x][y][i] = (1.0-alpha)*w[x][y][i];
		}
		sum += w[x][y][i];
	}

	//normalize
	for(int i = 0 ; i < k ; ++i) {
		w[x][y][i] /= sum;
	}
}

bool fitGaussian(int x, int y, Vec3b pix) {

	bool foundMatch = false;
	int foundNum = 0;

	uchar blue = pix.val[0];
	uchar green = pix.val[1];
	uchar red = pix.val[2];

	
	
	for(int i = 0 ; i < k ; ++i) {
		if(match(blue,muBlue[x][y][i],sigma[x][y][i]) && match(green,muGreen[x][y][i],sigma[x][y][i]) && match(red,muRed[x][y][i],sigma[x][y][i])) {
			foundNum = i;
			foundMatch = true;
			updateWeights(x, y, i);
			sortByWeights(x, y);
			float rho=alpha*(1.0/(pow (2.0*M_PI*sigma[x][y][i]*sigma[x][y][i], 1.5)))*exp(-0.5*(pow(((float)blue - muBlue[x][y][i]), 2.0) + pow(((float)green - muGreen[x][y][i]), 2.0) + pow(((float)red - muRed[x][y][i]), 2.0))/pow(sigma[x][y][i], 2.0));
			
			updateMuSigma(x, y, i, rho, (int)blue, (int)green, (int)red);
			
			break;
		}
	}
	
	if(!foundMatch) {
		w[x][y][k-1] = 0.33/((float)k);
		muBlue[x][y][k-1] = (float)blue;
		muGreen[x][y][k-1] = (float)green;
		muRed[x][y][k-1] = (float)red;
		sigma[x][y][k-1] = 6.0;
		
		updateWeights(x,y,k-1);
	}
	
	return foundMatch;
}


bool match(int xVal, float mu, float sigma) {
	if(fabs(xVal-mu)<=2.5*sigma) {
		return true;
	} else {
		return false;
	}
}

void updateMuSigma(int x, int y, int i, float rho, int blue, int green, int red) {
	muBlue[x][y][i] = (1.0-rho)*muBlue[x][y][i] + rho*blue;
	muGreen[x][y][i] = (1.0-rho)*muGreen[x][y][i] + rho*green;
	muRed[x][y][i] = (1.0-rho)*muRed[x][y][i] + rho*red;
	
	sigma[x][y][i] = sqrt((1.0-rho)*pow(sigma[x][y][i], 2.0) + (rho)*(pow(((float)blue - muBlue[x][y][i]),2.0) + pow(((float)green - muGreen[x][y][i]),2.0) + pow(((float)red - muRed[x][y][i]),2.0)));

}

void updateBackgroundForegroundModel(int y, int x) {
	float wSum = 0;
	int i=0;
	float bVal = 0;
	float gVal = 0;
	float rVal = 0;
				
	//mean of first B gaussians		
	do{
		bVal += w[x][y][i]*muBlue[x][y][i];
		gVal += w[x][y][i]*muGreen[x][y][i];
		rVal += w[x][y][i]*muRed[x][y][i];
					
		wSum += w[x][y][i];
		i++;
	} while (wSum < weightThresh);
	
	bVal /= wSum;
	gVal /= wSum;
	rVal /= wSum;
				
	bg.at<Vec3b>(y,x)[0] = bVal;
	bg.at<Vec3b>(y,x)[1] = gVal;
	bg.at<Vec3b>(y,x)[2] = rVal;

	if(fabs(frame.at<Vec3b>(y,x)[0] - bVal )< 30 &&fabs( frame.at<Vec3b>(y,x)[1] - gVal) < 30 && fabs(frame.at<Vec3b>(y,x)[2] - rVal) < 30) {
		fore.at<uchar>(y,x) = 0;
	} else {
		fore.at<uchar>(y,x) = 255;
	}
}


void sortByWeights(int x, int y) {
	//using n^2 sorting here
	
	for(int i = 1 ; i < k ; ++i) {
		for(int j = 0 ; j < k-i ; ++j) {
			if(w[x][y][j]/sigma[x][y][j] < w[x][y][j+1]/sigma[x][y][j+1]) {
				float temp = w[x][y][j];
				w[x][y][j] = w[x][y][j+1];
				w[x][y][j+1]= temp;
								
				temp = muBlue[x][y][j];
				muBlue[x][y][j] = muBlue[x][y][j+1];
				muBlue[x][y][j+1]= temp;
				
				temp = muGreen[x][y][j];
				muGreen[x][y][j] = muGreen[x][y][j+1];
				muGreen[x][y][j+1]= temp;
				
				temp = muRed[x][y][j];
				muRed[x][y][j] = muRed[x][y][j+1];
				muRed[x][y][j+1]= temp;
				
				temp = sigma[x][y][j];
				sigma[x][y][j] = sigma[x][y][j+1];
				sigma[x][y][j+1]= temp;
			}
		}
	}
}





