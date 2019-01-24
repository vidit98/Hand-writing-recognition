#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "algorithm"

using namespace std;
using namespace cv;

namespace bv
{
	CV_EXPORTS_W Mat preprocess1(Mat mat); //orient and remove line
	CV_EXPORTS_W Mat preprocess2(Mat mat); // kfill 
	CV_EXPORTS_W vector <vector<Rect> > preprocess3(Mat mat); // boundboxes
	CV_EXPORTS_W vector<vector<vector<int> > > preprocess4(); // group words
	
}

