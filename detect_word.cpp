#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "algorithm"

using namespace std;
using namespace cv;

#define threshold_length 20

Mat img = imread("handwriting6.jpg", 0);
Mat adapt_img, copied;

int is_valid(int x, int y){

	if(x>=0 && x<img.rows && y>=0 && y<img.cols)
		return 1;
	else
		return 0;
}

int extract_continous_patt(int x, int y, int* center, vector<Point> &points){

	center[0] += y;
	center[1] += x;
	points.push_back(Point(y, x));
	
	int len = 0;
	for(int i=-1; i<=1; i++){
		for(int j=-1; j<=1; j++){
			if(is_valid(x+i, y+j)){
				if(adapt_img.at<uchar>(x+i,y+j) == 255 && copied.at<uchar>(x+i,y+j) == 255){
					copied.at<uchar>(x+i,y+j) = 0;
					len += 1;
					len += extract_continous_patt(x+i, y+j, center, points);
				}
			}

		}
	}
	return len;
}
int main(){

	Mat img2 = imread("handwriting6.jpg", 1);
	Mat dst;
	GaussianBlur( img, img, Size(5, 5), 0, 0);

	adaptiveThreshold(img, adapt_img, 255, BORDER_REPLICATE, THRESH_BINARY_INV, 11, 1);
	medianBlur(adapt_img, adapt_img, 5);
	copied = adapt_img.clone();

	vector< vector<Point>> v;
	vector<Point> cent_pts;
	vector<Rect> boundRect;

	int length =0, center[2];
	for(int i=0; i<adapt_img.rows; i++){
		for(int j=0;j<adapt_img.cols;j++){
			if(adapt_img.at<uchar>(i,j) == 255 && copied.at<uchar>(i,j) == 255){
				vector<Point> temp;
				center[0] = center[1] = 0;
				copied.at<uchar>(i,j) = 0;
				length = extract_continous_patt(i, j, center, temp);
				if(length > threshold_length){
					v.push_back(temp);
					Point cent(int(center[0]/length), int(center[1]/length));
					cent_pts.push_back(cent);
					//circle(img2, cent, 5, Scalar(0,0,255),-1);
				}
			}
		}
	}
	
	int avg_height = 0;

	for(int i=0; i<v.size(); i++){
		boundRect.push_back(boundingRect(Mat(v[i])));
		//rectangle( img2, boundRect[i].tl(), boundRect[i].br(), Scalar(0,255,0), 2, 8, 0 );
		avg_height += boundRect[i].br().y - boundRect[i].tl().y; 
	}	

	avg_height /= v.size(); 
	cout<<"average height is "<<avg_height<<endl;

	vector<int> new_v1;
	vector<int> new_v2;
	//vector<vector<Point> > new_v3;

	for(int i=0; i<boundRect.size(); i++){
		cout<<i<<" "<<boundRect.size()<<endl;
		int h = boundRect[i].br().y - boundRect[i].tl().y;
		int w = boundRect[i].br().x - boundRect[i].tl().x;

		if(h>0.5*avg_height && h<2*avg_height && w > 0.4*avg_height){
			new_v1.push_back(i);
			rectangle( img2, boundRect[i].tl(), boundRect[i].br(), Scalar(0,255,0), 2, 8, 0 );
		}
		else if(h>2*avg_height && w>0.4*avg_height){
			//new_v2.push_back(i);
			new_v1.push_back(i);
			rectangle( img2, boundRect[i].tl(), boundRect[i].br(), Scalar(255,0,0), 2, 8, 0 );
			int no_of_step = h/avg_height;
			int width = boundRect[i].br().x - boundRect[i].tl().x, height;
			for(int l=1;l<no_of_step;l++){
				if(l!=(no_of_step-1))
					height = avg_height;
				else
					height = boundRect[i].br().y - boundRect[i].tl().y - l*avg_height;
				
				int top_y = boundRect[i].tl().y + l*avg_height;
				int bot_y = boundRect[i].tl().y + l*avg_height + height;
				vector<Point> tt;
				center[0] = center[1] = 0;
				int iterations = 0;
				for(int k=v[i].size()-1;k>=0;k--){
					if(v[i][k].y >= top_y && v[i][k].y <= bot_y){
						tt.push_back(v[i][k]);
						center[0] += v[i][k].x;
						center[1] += v[i][k].y;
						v[i].erase(v[i].begin() + k);
						iterations++;
					}
				}

				if(tt.size()>0){
					v.push_back(tt);
					cent_pts.push_back(Point(center[0]/iterations, center[1]/iterations));
					//boundRect.push_back(Rect(boundRect[i].tl().x, boundRect[i].tl().y+l*avg_height, width, height));
					boundRect.push_back(boundingRect(Mat(tt)));
				}
				else
					cout<<"something is wrong\n";

			}
			center[0] = center[1] = 0;
			int iterations = 0;
			for(int l=0;l<v[i].size();l++){
				center[0] += v[i][l].x;
				center[1] += v[i][l].y;
				iterations++;
			}
			center[0] /= iterations;
			center[1] /= iterations;

			cent_pts[i] = Point(center[0], center[1]);
			boundRect[i] = boundingRect(Mat(v[i]));
		}

	}
	cout<<"jjjjjjjjj\n";
	vector<vector<Point> > extra_v1;

	for(int i=0;i<new_v1.size();i++){

		vector<Point> temp;
		int step_size = max(int((boundRect[new_v1[i]].br().x-boundRect[new_v1[i]].tl().x)/avg_height),1);
		int small_rect_w = avg_height;
		int small_rect_h = boundRect[new_v1[i]].br().y - boundRect[new_v1[i]].tl().y;

		for(int j=0;j<step_size;j++){
			if(boundRect[new_v1[i]].tl().x+j*small_rect_w + small_rect_w < adapt_img.cols){
				Mat roi = adapt_img(Rect(boundRect[new_v1[i]].tl().x+j*small_rect_w, boundRect[new_v1[i]].tl().y, small_rect_w, small_rect_h));
				int center_x = 0, center_y = 0, no = 0;

				for(int k=0;k<roi.rows;k++){
					for(int l=0;l<roi.cols;l++){
						if(roi.at<uchar>(k,l) == 255){
							center_x += l;
							center_y += k;
							no++;
						}
					}
				}
				if(no != 0){
					center_y /= no;
					center_x /= no;
					Point p = Point(boundRect[new_v1[i]].tl().x+j*small_rect_w + center_x, boundRect[new_v1[i]].tl().y + center_y);
					circle(img2, p, 5, Scalar(255,0,255),-1);
					temp.push_back(p);
				}
				else
					continue;
			}
		}
		extra_v1.push_back(temp);
	}
	cout<<"jjjjjjjjj\n";
	// HOUGH TRANSFORM 

	int d = sqrt(adapt_img.rows*adapt_img.rows + adapt_img.cols*adapt_img.cols);
	Mat img3(2*(d+2), 20, CV_8UC1, Scalar(0));

	for (int i = 0; i < extra_v1.size(); i++){
		for (int j = 0; j < extra_v1[i].size(); j++){
			
			int x = extra_v1[i][j].x;
			int y = extra_v1[i][j].y;

			for (int th=80; th<100; th++){
				float angle = 3.14*th/180;
				int l = x*cos(angle) + y*sin(angle);
				if (img3.at<uchar>(d+2+l,th) < 255)
					img3.at<uchar>(d+2+l,th)+=2;

			}
			
		}
	}

	// DETECTING THE LINES USING HOUGH TRANSFORM

	int flag  = 1;
	double maxx, minn;
	Point max_p, min_p;
	int iterations = 0;
	float line_threshold = 0;
	vector<Vec3f> lines;

	vector<vector<Point> > temp_pts = extra_v1;

	while(flag){

		minMaxLoc(img3, &minn, &maxx, &min_p, &max_p);
		int a = img3.at<uchar>(max_p.y,max_p.x);

		if((iterations != 0 && a < int(0.7*line_threshold/iterations)) || temp_pts.size()==0)
			break;
			
		int theta = max_p.x;
		int dist = max_p.y-d-2;
		
		line_threshold += img3.at<uchar>(max_p.y,max_p.x);

		vector<int> tempp;
		int tot_pts = 0;
		for(int i=temp_pts.size()-1;i>=0;i--){

			int tot = 0;
			for(int j=0;j<temp_pts[i].size();j++){
				float angle = (theta+80)*3.14/180;
				int l = temp_pts[i][j].x*cos(angle) + temp_pts[i][j].y*sin(angle);
				if(abs(l-dist)<=avg_height){
					tot++;
				}
			}
			//cout<<tot<<endl;
			if(tot>int(temp_pts[i].size()/2)){
				tot_pts++;
				for(int j=0;j<temp_pts[i].size();j++){
					for (int th=80; th<100; th++){
						float angle = 3.14*th/180;
						int l = temp_pts[i][j].x*cos(angle) + temp_pts[i][j].y*sin(angle);
						if (img3.at<uchar>(d+2+l,th) > 2)
							img3.at<uchar>(d+2+l,th)-=2;

					}
				}
				temp_pts.erase(temp_pts.begin() + i);
			}

		}
		
		iterations++;
		lines.push_back(Vec3f(theta, dist, tot_pts));
	}

	cout<<iterations<<" "<<temp_pts.size()<<endl;
	
	sort(lines.begin(), lines.end(), 
		  [](const cv::Vec3f &a, const cv::Vec3f &b)
		  {
			  return a[1] < b[1]; 
		  });

	float avg_line_dist = 0;
	vector<float> dis;

	for(int i=0;i<lines.size()-1;i++){
		float angle1 = (lines[i][0]+80)*3.14/180;
		float angle2 = (lines[i+1][0]+80)*3.14/180;
		float y1 = lines[i][1]/sin(angle1) - adapt_img.cols/(tan(angle1)*2.0);
		float y2 = lines[i+1][1]/sin(angle2) - adapt_img.cols/(tan(angle2)*2.0);
		//circle(img2, Point(int(adapt_img.cols/2),int(y1)) , 8, Scalar(100,127,255),-1);
		//circle(img2, Point(int(adapt_img.cols/2),int(y2)) , 8, Scalar(100,127,255),-1);
		dis.push_back(fabs(y2 - y1));	
	}

	sort(dis.begin(),dis.end());
	avg_line_dist = dis[int(dis.size()/2)];
	cout<<avg_line_dist<<endl;
	dis.erase(dis.begin(), dis.end());

	//REMOVING EXTRA LINE
	int threshold = 0.5*avg_line_dist;
	if(threshold<avg_height)
		threshold = avg_height;

	for(int i=lines.size()-2;i>=0;i--){
		
		float angle1 = (lines[i][0]+80)*3.14/180;
		float angle2 = (lines[i+1][0]+80)*3.14/180;
		float y1 = lines[i][1]/sin(angle1) - adapt_img.cols/(tan(angle1)*2.0);
		float y2 = lines[i+1][1]/sin(angle2) - adapt_img.cols/(tan(angle2)*2.0);
		if( fabs(y2-y1) < threshold){
			if(lines[i][2] <= lines[i+1][2])
				lines.erase(lines.begin()+i);
			else
				lines.erase(lines.begin()+i+1);
		}
	}
	
	vector<vector<int > >line_assigned(lines.size());
	for(int i=0;i<new_v1.size();i++){

		float min = 10000;
		int min_index = -1;
		for(int j=0;j<lines.size();j++){

			float angle = (lines[j][0] + 80)*3.14/180;
			float dist = fabs(cent_pts[new_v1[i]].x*cos(angle) + cent_pts[new_v1[i]].y*sin(angle) - lines[j][1]); 
			if(dist<min){
				min = dist;
				min_index = j;
			}
		}

		if(min<0.9*avg_line_dist){
			line_assigned[min_index].push_back(new_v1[i]);
		}

	}

	for(int i=0;i<line_assigned.size();i++){

		vector<float> slopes;

		for(int j=0;j<line_assigned[i].size()-1;j++){
			int diff_y = cent_pts[line_assigned[i][j]].y-cent_pts[line_assigned[i][j+1]].y;
			int diff_x = cent_pts[line_assigned[i][j]].x-cent_pts[line_assigned[i][j+1]].x;
			float s = atan2(diff_y, diff_x)*180.0/3.14;
			
			if(s>90)
				s = s - 180;
			else if(s<-90)
				s = s + 180;
			s = s + 90;
			slopes.push_back(s);
						
		}

		sort(slopes.begin(), slopes.end());	
		if(slopes.size()>0){	
			float correct_slope = slopes[int(slopes.size()/2)]; 
			
			cout<<correct_slope<<" "<<lines[i][0]+80<<endl;
			vector<float> perpendicular;

			for(int j=0;j<line_assigned[i].size();j++){
				float ang = correct_slope*3.14/180;
				Point p(cent_pts[line_assigned[i][j]].x,cent_pts[line_assigned[i][j]].y);
				float diff = p.x*cos(ang) + p.y*sin(ang)-lines[i][1];
				perpendicular.push_back(diff);
				cout<<diff<<" "; 	
			}

			cout<<endl;

			if(correct_slope>=85 && correct_slope<=95){
				sort(perpendicular.begin(), perpendicular.end());
				lines[i][0] = correct_slope - 80;
				lines[i][1] += perpendicular[(int(perpendicular.size()/2))%perpendicular.size()];
			}
			line_assigned[i].clear();
		}
	}
	

	//REMOVING THE EXTRA LINE ONCE AGAIN
	sort(lines.begin(), lines.end(), 
		  [](const cv::Vec3f &a, const cv::Vec3f &b)
		  {
			  return a[1] < b[1]; 
		  });

	for(int i=lines.size()-2;i>=0;i--){
		float angle1 = (lines[i][0]+80)*3.14/180;
		float y1 = lines[i][1]/sin(angle1) - adapt_img.cols/(tan(angle1)*2.0);

		for(int j=1;j<4;j++){
			if(i+j<lines.size()){
				float angle2 = (lines[i+j][0]+80)*3.14/180;
				float y2 = lines[i+j][1]/sin(angle2) - adapt_img.cols/(tan(angle2)*2.0);
				if( fabs(y2-y1) < threshold){
					if(lines[i][2] <= lines[i+j][2])
						lines.erase(lines.begin()+i);
					else
						lines.erase(lines.begin()+i+j);
				}
			}
		}
	}
	

	for(int i=0; i<lines.size(); i++){
			
		int rho = lines[i][1]; 
		float theta = (lines[i][0] + 80)*3.14/180;
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		pt1.x = cvRound(x0 + 1000*(-b));
		pt1.y = cvRound(y0 + 1000*(a));
		pt2.x = cvRound(x0 - 1000*(-b));
		pt2.y = cvRound(y0 - 1000*(a));
		line( img2, pt1, pt2, Scalar(0,0,255), 3, CV_AA);	
	}
	cout<<endl<<lines.size();

	for(int i=0;i<new_v1.size();i++){

		float min = 10000;
		int min_index = -1;
		for(int j=0;j<lines.size();j++){

			float angle = (lines[j][0] + 80)*3.14/180;
			float dist = fabs(cent_pts[new_v1[i]].x*cos(angle) + cent_pts[new_v1[i]].y*sin(angle) - lines[j][1]); 
			if(dist<min){
				min = dist;
				min_index = j;
			}
		}

		if(min<0.9*avg_line_dist){
			for(int l=0;l<v[new_v1[i]].size();l++){
				img2.at<Vec3b>(v[new_v1[i]][l].y, v[new_v1[i]][l].x) = {30+50*min_index, 50+60*min_index, 20+80*min_index};
			}
		}
	}
	cout<<"threshold is "<<avg_height/2<<endl;

	imwrite("adaptive2.jpg", img2);
	//imwrite("hough.jpg", img3);
	waitKey(0);
				
}