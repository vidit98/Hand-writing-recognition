// https://hal.inria.fr/hal-00799331v2/document
#include "wrapper.hpp"
namespace bv
{
#define threshold_length 20
#define kfill_eq(c, n, k, r) ((c == 1) && ((n > 3*k - 4) || ((n == 3*k - 4) && (r == 2))))



int is_valid(int x, int y){

	if(x>=0 && x<2339 && y>=0 && y<1654)
		return 1;
	else
		return 0;
}

float find_median(float ar[],int size)
{
	float m;
	sort(ar , ar + size);
	if(size % 2 == 0)
	{
		m= (ar[size/2] + ar[(size/2 -1)])/2 ;
	}
	else
	{
		m= ar[size/2];
	}
	return m;

}

float cal_slp(int x1,int y1,int x2,int y2)
{

	float m;
	if (x1==x2) return CV_PI/2;
	m=atan((float)(y1-y2)/(float)(x1-x2));
	return m;

}

void orient(Mat img, Mat &out)
{
	vector<Vec4i> lines;
	Mat img1;
	Mat vis =  imread("pdftoimage/FORM - FREE TEXT INOUT BOXES_3-1.jpg", 1);
	threshold(img, img1, 128, 255, THRESH_BINARY_INV|THRESH_OTSU);
	HoughLinesP(img1,lines,1,CV_PI/180,50,80,1);
	
	float avgp[1000] = {} , avgn[1000] = {};
	int np = 0, nn = 0;

	
	for( size_t i = 0; i < lines.size(); i++ )
    {
    	
    	float slp = cal_slp(lines[i][0],-lines[i][1],lines[i][2],-lines[i][3]);
        
        if(slp < 0)
        {
        	line( vis, Point(lines[i][0], lines[i][1]),
	            Point(lines[i][2], lines[i][3]), Scalar(0,0,255), 1, 8 );
        	avgn[nn++]=slp;
        	
        }
        else if(slp >= 0)
        {
        	avgp[np++]=slp;
        }

    }

    int var = max(np,nn);

    if (var < 10)
	{
		out = img.clone();
		cout << "NOT RUNNING";
		return;
	}

    imwrite("linee.jpg", vis);
    
    
    cout << "Numbers" << nn << " " << np << endl;
    
    
   
   
    Point2f center((img.cols-1)/2.0, (img.rows-1)/2.0);
    Rect bbox;
    Mat rot;

   
    if ( nn > np)
    {
    	float med_n = find_median(avgn, nn);
    	med_n= (med_n*180)/(CV_PI);
    	rot = getRotationMatrix2D(center, -med_n, 1.0);
    	bbox = RotatedRect(Point2f(), img.size(), -med_n).boundingRect();
    	cout << endl << med_n << "med n";

		
    }
    else if (np >= nn)
    {
    	float med_p =  find_median(avgp, np);
    	med_p= (med_p*180)/(CV_PI);
    	rot = getRotationMatrix2D(center, -med_p, 1.0);
    	bbox = RotatedRect(Point2f(), img.size(), -med_p).boundingRect();
    	cout << med_p << "med p";
    }

   
    rot.at<double>(0,2) += bbox.width/2.0 - img.cols/2.0;
    rot.at<double>(1,2) += bbox.height/2.0 - img.rows/2.0;
    

    warpAffine(img, out, rot, bbox.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(255));
    
    printf("%s\n","qqqqqqqqqqqqqqqqqqqqqqq" );
}

void remove_lines(Mat img, Mat &out)
{
	Mat img_bin, temp1, vertical_img, horizon_img;
	int kernel_length = img.cols/80;
	cout << kernel_length;
	adaptiveThreshold(img, img_bin, 255, BORDER_REPLICATE, THRESH_BINARY_INV, 11, 1);

	Mat vertical = getStructuringElement(MORPH_RECT, Size(1, kernel_length + 5));
	Mat horizon = getStructuringElement(MORPH_RECT, Size(kernel_length, 1));

	// Morphological operation to detect vertical lines from an image

	erode(img_bin, temp1, vertical, Point(-1,-1),3);
	dilate(temp1, vertical_img, vertical, Point(-1,-1),3);

	erode(img_bin, temp1, horizon, Point(-1,-1),2);
	dilate(temp1, horizon_img, horizon, Point(-1,-1),3);

	bitwise_or(vertical_img, horizon_img, temp1);

	bitwise_or(temp1, img, out);
	printf("%s\n", "Lines removed");

}

void kfill(Mat img, Mat &output, int k_size, int even) // even is 1 when k_size is even
{

	int thr = (k_size-2)*(k_size-2)/2;
	int size = 4*(k_size-1);
	Mat img1 = img.clone();
	int val = k_size/2 - even;
	// cout <<"Colorrrrrrrrrrrrr" << int(img.at<uchar>(29, 10)) << " " ;

	for (int c = val; c < img.cols - k_size/2; c++)
	{
		for (int r = val; r < img.rows - k_size/2; r++)
		{
			int on = 0;
			int connect[size];// = {};
			for(int i=0;i<size;i++)
				connect[i] = 0;
			
			for (int i = -(val) ; i <= k_size/2 ; ++i)
			{
				for (int j = -(val); j <= k_size/2 ; ++j)
				{	
					
					if(i == -val)
					{
						if (int(img.at<uchar>(r + i, j + c)) == 255)
							connect[j+val] = 1;
						// cout << "1st " << j + val << endl;
						continue;
					}
					else if(i == k_size/2)
					{
						if (int(img.at<uchar>(r + i, j + c)) == 255)
							connect[size - k_size + 2  - (j+val) - 1] = 1;
						// cout << "2nd " <<size - k_size + 2  - (j+val) - 1 << endl;
						continue; 
					}
					else if(j == -val)
					{
						if (int(img.at<uchar>(r + i, j + c)) == 255)
							connect[size - (i + val)] = 1;
						// cout << "3rd " << size - (i + val) << endl;
						continue;
					}
					else if(j == k_size/2)
					{
						if (int(img.at<uchar>(r + i, j + c)) == 255)
							connect[k_size - 1 + i + val] = 1;
						// cout << "4th " << k_size - 1 + i + val << endl;
						continue;
					}

					if(int(img.at<uchar>(r + i, j + c)) == 255)
						on++;
				}
			}
			
			
			// cout << "ihihoih";
			int conn_on = 0;
			int conn_off = 0;
			int bound_on =  0;
			int corner_on = 0;
			int corner_off = 0;


			for (int a = 0; a < size; ++a)
			{
				// cout << connect[a] << " " ;
				if(connect[a] < connect[(a+1)%(size-1)])
				{
					conn_off += 1;
				}
				else if (connect[a] > connect[(a+1)%(size-1)])
				{
					conn_on += 1;
				}
				if (connect[a])
				{
					bound_on++;	
				}
				if (a%(k_size-1) == 0)
				{	if(connect[a] == 1)
					{
						corner_on +=1;
					}
					else if(connect[a] == 0)
					{
						corner_off += 1;
					}
				}		

			}
			
			if (on >= thr)
			{
				// printf("%d %d %d %d\n", conn_off, size-on, k_size, corner_off );
				// cout << kfill_eq(conn_off, size-on, k_size, corner_off) << endl;
				if (kfill_eq(conn_off, size-bound_on, k_size, corner_off))
				{
					for (int i = -(val) + 1 ; i <= k_size/2 -1 ; ++i)
					{
						for (int j = -(val) + 1; j <= k_size/2 - 1 ; ++j)
						{
							output.at<uchar>(r+i,c+j) = 0;
						}
					}
				}
				else
				{
					for (int i = -(val) + 1 ; i <= k_size/2 -1 ; ++i)
					{
						for (int j = -(val) + 1; j <= k_size/2 - 1 ; ++j)
						{
							output.at<uchar>(r+i,c+j) = 255;
							// cout << "filling output at:" << r+i << " " << c+j << endl;

						}
					}	
				}
			}
			else
			{
				if (kfill_eq(conn_on, bound_on, k_size, corner_on))
				{
					for (int i = -(val) + 1 ; i <= k_size/2 -1 ; ++i)
					{
						for (int j = -(val) + 1; j <= k_size/2 - 1 ; ++j)
						{
							output.at<uchar>(r+i,c+j) = 255;
						}
					}
				}
				else
				{
					for (int i = -(val) + 1 ; i <= k_size/2 -1 ; ++i)
					{
						for (int j = -(val) + 1; j <= k_size/2 - 1 ; ++j)
						{
							output.at<uchar>(r+i,c+j) = 0;
						}
					}	
				}
			}
			// /rectangle( output, Point(c-val, r -val),Point(c+k_size/2, r + k_size/2), Scalar(0,255,0), 1, 8, 0 );
		

		}
	}

	printf("%s\n","exitttt" );
}
int extract_continous_patt(Mat filtered, Mat &copied, int x, int y, int* center, vector<Point> &points){

	center[0] += y;
	center[1] += x;
	points.push_back(Point(y, x));
	
	int len = 0;
	for(int i=-1; i<=1; i++){
		for(int j=-1; j<=1; j++){
			if(is_valid(x+i, y+j)){
				// cout << x+i << " " << y+j << endl;
				if(filtered.at<uchar>(x+i,y+j) == 0 && copied.at<uchar>(x+i,y+j) == 0){
					copied.at<uchar>(x+i,y+j) = 255;
					len += 1;
					len += extract_continous_patt(filtered, copied, x+i, y+j, center, points);
				}
			}

		}
	}
	return len;
}

void getBoundingBoxes(Mat img, vector<Rect> &boundRect, vector<Point> &lower_mid)
{
	Mat copied = img.clone();
	Mat dst0 = img.clone();
	vector< vector<Point> > v;
	vector<Point> cent_pts;
	vector<Rect> tempRect;



	int length =0, center[2];
	for(int i=0; i<img.rows; i++){
		for(int j=0;j<img.cols;j++){

			// cout << i << " " << j << endl; 
			if(img.at<uchar>(i,j) == 0 && copied.at<uchar>(i,j) == 0){
				vector<Point> temp;
				center[0] = center[1] = 255;
				copied.at<uchar>(i,j) = 255;
				
				length = extract_continous_patt(img, copied,i, j, center, temp);
				
				if(length > threshold_length){
				
					v.push_back(temp);
					Point cent(int(center[0]/length), int(center[1]/length));
					cent_pts.push_back(cent);
				}
			}
		}
	}

	int avg_height = 0, max_w = 0, max_h = 0;
	int width[img.cols];// = {};
	int ht[img.rows];// = {};
	for(int i=0;i<img.cols;i++)
		width[i] = 0;
	for (int i = 0; i < img.rows; i++)
		ht[i] = 0;
	
	printf("%s\n","Done" );

	for(int i=0; i<v.size(); i++){

		tempRect.push_back(boundingRect(Mat(v[i])));
		rectangle( dst0, tempRect[i].tl(), tempRect[i].br(), Scalar(0,255,0), 2, 8, 0 );
		width[tempRect[i].br().x - tempRect[i].tl().x] += 1;
		ht[tempRect[i].br().y - tempRect[i].tl().y] += 1 ;

		if(tempRect[i].br().x - tempRect[i].tl().x > max_w)
			max_w = tempRect[i].br().x - tempRect[i].tl().x;
		if(tempRect[i].br().y - tempRect[i].tl().y > max_h)
			max_h = tempRect[i].br().y - tempRect[i].tl().y;
	}

	printf("%s\n","Done" );
	imwrite("rec.jpg", dst0);

	cout << max_h << " " << max_w << endl;
	int mod_x = distance(width, max_element(width, width + max_w));
	int mod_y = distance(ht, max_element(ht, ht + max_h));

	cout << width[mod_x] << " " << ht[mod_y] << endl;

	cout << "mod_y" << mod_y << " " << "mod_x" << mod_x;



	for(int i=0; i<tempRect.size(); i++){
		// cout<<i<<" "<<boundRect.size()<<endl;
		int h = tempRect[i].br().y - tempRect[i].tl().y;
		int w = tempRect[i].br().x - tempRect[i].tl().x;
		int asp_rat = h/w;
		int area = tempRect[i].area();

		if(h < 100 && h >= 10 && asp_rat < 10 && area >= 100) // HARDCODED TO BE CHANGED
		{
			boundRect.push_back(tempRect[i]);
			lower_mid.push_back(Point(tempRect[i].tl().x, tempRect[i].br().y ));

		}
	}
}


bool compimp(Rect a , Rect b) { return a.tl().x  < b.tl().x;}
bool compimp1(Point a , Point b) { return a.x  < b.x;}
Point2f getline(int x1, int y1, int x2, int y2)
{
	Point2f pt;
	pt.x = float(y2 - y1)/(x2 - x1);
	pt.y = float(y1*x2 - y2*x1)/(x2 - x1);

	return pt;
}

float dis(Point2f pt, int x, int y){ return float(abs(y - (pt.x)*x - pt.y))/(1 + (pt.x)*(pt.x));}

void improvedBB(vector<Rect> boundRect, vector<Point> &initial_point, vector< vector< Rect> > &grp)
{
	Mat img1;
	// cvtColor(filtered.clone(), img1, COLOR_GRAY2RGB);

	sort(boundRect.begin(), boundRect.end(), compimp);
	Point2f p;
	vector<Point2f> lines;
	// vector<Point> initial_point;
	vector<Rect> v;
	int new_line_threshold = 10;

	// Initializing params
	grp.push_back(v);
	grp[0].push_back(boundRect[0]);
	initial_point.push_back(Point(boundRect[0].tl().x, boundRect[0].br().y));
	p.x = 0; p.y = -boundRect[0].br().y;
	lines.push_back(p);
	
	float min_d = 1000, d;
	int min_idx = 0;
	float avg_slope = 0;
	for (int i = 1; i < boundRect.size(); ++i)
	{
		// cvtColor(filtered.clone(), img1, COLOR_GRAY2RGB);
		float h1 = boundRect[i-1].br().y - boundRect[i-1].tl().y;
		float h2 = boundRect[i].br().y - boundRect[i].tl().y;
		// cout << "Print " << i << " " <<boundRect[i-1].tl().y << " " << boundRect[i].tl().y << " "  << boundRect[i-1].br().y << " " << boundRect[i].br().y << endl;
		float overlap = min(boundRect[i].br().y, boundRect[i-1].br().y) - max(boundRect[i-1].tl().y, boundRect[i].tl().y);

		float h = max(h1, h2)/min(h1, h2);
		
		
		

		if (h < 5 && overlap >= 0.5*(min(h1, h2)))
		{
			cout << "Pushed-------" << i << "in" << min_idx << endl;
			grp[min_idx].push_back(boundRect[i]);
		}
		else
		{	
			printf("%s\n","InElse" );
			min_d = 1000;
			for (int k = 0; k < lines.size(); ++k)
			{
				d = dis(lines[k], boundRect[i].br().x, -boundRect[i].br().y);

				printf("%d %f %f %f\n",k,d, lines[k].x, -lines[k].y);
				// circle(img1, Point(bottom_left[i].x,bottom_left[i].y), 2, Scalar(0,255,255),-1);
				// line( img1, initial_point[k], grp[k].back().br(), Scalar(0,255,0), 1, CV_AA);	
				
				if (d < min_d)
				{
					min_d = d;
					min_idx = k;
				}
			}

			Rect r =  grp[min_idx].back();
			
			h1 = r.br().y - r.tl().y;
			overlap = min(boundRect[i].br().y, r.br().y) - max(r.tl().y, boundRect[i].tl().y);
			h =  max(h1, h2)/min(h1, h2);
			cout << "Print " << i << " " <<r.tl().y << " " << boundRect[i].tl().y << " "  << r.br().y << " " << boundRect[i].br().y << endl;
			printf("%s %f %f %f %f\n","Data", h1, h2, h, overlap );
			int flag = 0;
			if (-r.br().x + boundRect[i].tl().x >= 10)
			{
				if (min_d <= new_line_threshold )
				{
					printf("%s %f\n", "COOOOOOOOOOOO||OOOOOOOOl", min_d);
					flag =1 ;
				}

				// Algo is based on neighbourhood properties therefore start new line if nex bounding box is far away from the previos
				if(-r.br().x + boundRect[i].tl().x >= 200)
					flag = 0;
			}
			else
			{
				if (h < 3 && overlap >= 0.4*(min(h1, h2)))
				{
					flag = 1;
				}
			}


			if (flag == 1)
			{
				printf("%s \n","Updated Line");
				cout << "Pushed-------" << i << "in" << min_idx << endl;
				
				printf("%s %d %d %d %d\n","Points for line" ,initial_point[min_idx].x, -initial_point[min_idx].y, boundRect[i].tl().x, -boundRect[i].br().y);
				p = getline(initial_point[min_idx].x, -initial_point[min_idx].y, boundRect[i].tl().x, -boundRect[i].br().y);
				p.x = ((lines[min_idx].x)*(grp[min_idx].size()) + p.x)/(grp[min_idx].size() + 1);
				p.y = ((lines[min_idx].y)*(grp[min_idx].size()) + p.y)/(grp[min_idx].size() + 1);
				lines[min_idx] = p;

				grp[min_idx].push_back(boundRect[i]);
			}
			else
			{
				grp.push_back(v);
				printf("%s %d\n", "New line generated", grp.size() - 1);
				
				min_idx = grp.size() - 1;
				grp[min_idx].push_back(boundRect[i]);
				initial_point.push_back(Point(boundRect[i].tl().x, boundRect[i].br().y));
				p.x = 0; p.y = -boundRect[i].br().y;
				lines.push_back(p);
				
			}
		}	
	}

	for (int k = 0; k < lines.size(); ++k)
	{
		avg_slope += lines[k].x;
		line( img1, initial_point[k], grp[k].back().br(), Scalar(0,0,255), 1, CV_AA);	
	}
	printf("%s %f\n","Avg slope", atan(avg_slope)*180/(CV_PI*lines.size()));
	imwrite("vis.jpg", img1);




}

void mergeLines(vector<Point> initial_point, vector< vector< Rect> > &grp)
{
	

	Mat img1, img2;
	// cvtColor(filtered.clone(), img1, COLOR_GRAY2RGB);cvtColor(filtered.clone(), img2, COLOR_GRAY2RGB);
	int count = 0;
	for (int i = 0;  i < initial_point.size() ; i++)
	{
		count ++;
		vector<int> v;
		v.push_back(i);
		// printf(" %s %d\n","Size",i);
		// cvtColor(filtered.clone(), img2, COLOR_GRAY2RGB);
		for (int j = 0; j < initial_point.size(); ++j)
		{
			
			
			// line( img1, initial_point[i],  grp[i].back().br(), Scalar(0,0,255), 1, CV_AA);
			if(i == j)
				continue;
			int start_dis = initial_point[j].x - initial_point[i].x;
			int end_dis = initial_point[j].x - grp[i].back().br().x;
			int start_dis1 = grp[j].back().br().x - initial_point[i].x;
			int end_dis1 = grp[j].back().br().x - grp[i].back().br().x;
			// cout << start_dis << " " << end_dis << " " << start_dis1 << " " << end_dis1 << endl;
			if (start_dis > 0)
			{
				continue;
			}
			// line( img2, initial_point[j],  grp[j].back().br(), Scalar(0,0,255), 1, CV_AA);
			int b = -1;
			float a = 100000;
			if (start_dis*start_dis1 < 0 || end_dis1*end_dis < 0)
			{
				Point2f l = getline(initial_point[j].x, -initial_point[j].y, grp[j].back().br().x, -grp[j].back().br().y );
				if (start_dis*start_dis1 < 0)
				{
					a = abs(dis(l, initial_point[i].x, -initial_point[i].y));
				}
				else if(end_dis1*end_dis < 0 &&  end_dis1*end_dis < 0)
				{
					a = min(abs(dis(l, initial_point[i].x, -initial_point[i].y)), abs(dis(l,grp[i].back().br().x, -grp[i].back().br().y)));
				}
				
			}
			b = min(abs(start_dis1), abs(start_dis));
			b = min(b, abs(end_dis));
			b = min(b,abs(end_dis1));
 			if(a < 7)
			{
				// line( img1, initial_point[i], initial_point[j], Scalar(0,255,0), 1, CV_AA);
				v.push_back(j);	
			}

			else if (b < 150)
			{
				
				if (b == abs(start_dis))
				{

					if (abs(initial_point[j].y - initial_point[i].y) <= 15)
					{
						// line( img1, initial_point[i], initial_point[j], Scalar(0,255,0), 1, CV_AA);
						v.push_back(j);
					}
				}
				else if (b == abs(start_dis1))
				{
					if (abs(grp[j].back().br().y - initial_point[i].y) <= 15 )
					{
						// line( img1, initial_point[i], initial_point[j], Scalar(0,255,0), 1, CV_AA);
						v.push_back(j);
					}
				}
				else if ( b == abs(end_dis) )
				{
					if (abs(grp[j].back().br().y - initial_point[i].y) <= 15)
					{
						// line( img1, initial_point[i], initial_point[j], Scalar(0,255,0), 1, CV_AA);
						v.push_back(j);
					}
				}
				else if(b == abs(end_dis1))
				{
					if (abs(grp[j].back().br().y - grp[i].back().br().y) <= 15 )
					{
						// line( img1, initial_point[j], grp[i].back().br(), Scalar(0,255,0), 1, CV_AA);
						v.push_back(j);
					}

				}
			}
		
		}
		

		for (int k = 1; k < v.size(); ++k)
		{
			if (v[k] < i)
				i--;
			int s = grp[v[k]].size();
			for (int q = 0; q < s; ++q)
			{
				if (grp[v[k]][q].br().x > grp[v[0]].back().br().x)
				{
					grp[v[0]].push_back(grp[v[k]][q]);
				}
				else
				{
					grp[v[0]].insert(grp[v[0]].begin(), grp[v[k]][q]);
				}

				
			}
		}
		v.erase(v.begin());
		sort(v.begin(), v.end(),greater<int>());
		for (int k = 0; k < v.size(); ++k)
		{
			grp.erase(grp.begin() + v[k]);
			initial_point.erase(initial_point.begin() + v[k]);
		}


	}

}

float perc_match(Mat filtered, Rect bound1, Rect bound2){

	Mat templ = filtered(bound1).clone();
	Mat src = filtered(bound2).clone();

	int resize_row = templ.rows, resize_cols = templ.cols;

	if(src.rows>templ.rows)
		resize_row = src.rows;

	if(src.cols>templ.cols)
		resize_cols = src.cols;

	resize(templ, templ, Size(resize_cols, resize_row));
	resize(src, src, Size(resize_cols, resize_row));

	int match = 0, t;

	for(int i=0;i<resize_row;i++){
		for(int j=0;j<resize_cols;j++){
			match += abs( templ.at<uchar>(i,j) - src.at<uchar>(i,j));
		}
	}
	


	return match/(resize_row*resize_cols*255.0);

}


void remove_I(Mat filtered, vector<Rect> &boundRect, vector<Rect> &rejected){

	vector<int> consider;

	for(int i=0;i<boundRect.size();i++){
		int width = boundRect[i].br().x - boundRect[i].tl().x;
		int hght =  boundRect[i].br().y - boundRect[i].tl().y;

		if(float(hght)/width >= 2.8 && width<=10){
			consider.push_back(i);
			cout<<hght<<" "<<width<<endl;
			
		}
	}

	vector<vector<int> > diff(1);
	vector<int> temp;
	temp.push_back(consider[0]);
	diff[0] = temp;

	for(int j=1;j<consider.size();j++){
		int flag = 0;
		for(int k=0;k<diff.size();k++){
			


			float perc = perc_match(filtered, boundRect[consider[j]], boundRect[diff[k][0]]);
			cout<<perc<<" ";
			if(perc<0.25){
				diff[k].push_back(consider[j]);
				flag = 1;
				//break;
			}
		}
		cout<<endl;

		if(!flag){
			temp.clear();
			temp.push_back(consider[j]);
			diff.push_back(temp);
		}
	}

	int max_amt = 0, ind = -1;
	for(int i=0;i<diff.size();i++){
		if(diff[i].size()>max_amt){
			max_amt = diff[i].size();
			ind = i;
		}
	}
	cout<<"total removed "<<max_amt<<endl;

	if(ind!=-1){

		
				sort(diff[ind].begin(), diff[ind].end());
				for(int k=diff[ind].size()-1;k>=0;k--){
					
					rejected.push_back(boundRect[diff[ind][k]]);
					boundRect.erase(boundRect.begin() + diff[ind][k]);
				}
		
		
	}

}

Rect large_box(vector<Rect> box, vector<int> nos){

	int top_x = box[nos[0]].tl().x , top_y=10000000, width, hgt= 0;

	width = box[nos[nos.size()-1]].br().x - box[nos[0]].tl().x;

	for(int i=0;i<nos.size();i++){

		int temp = box[nos[i]].br().y;
		if(temp>hgt)
			hgt = temp;

		int temp2 = box[nos[i]].tl().y;
		if(temp2 < top_y)
			top_y = temp2;	 

	} 

	hgt = hgt - top_y;

	Rect ans(top_x, top_y, width, hgt);

	return ans;
}

bool comp2(Rect a , Rect b) { return a.tl().x  < b.tl().x;}

void detect_word(vector<Rect> boxes, vector<Rect> &final_box){

	int dist[int(boxes.size())];
	sort(boxes.begin(), boxes.end(), comp2);

	for(int i=1;i<boxes.size();i++){

		//float min_dist = 100000;

		/*for(int j=0;j<boxes.size();j++){

			if(j!=i){
				
				Point cent1((boxes[i].br().x+boxes[i].tl().x)/2, (boxes[i].br().y+boxes[i].tl().y)/2);
				Point cent2((boxes[j].br().x+boxes[j].tl().x)/2, (boxes[j].br().y+boxes[j].tl().y)/2);
				float temp = sqrt(pow(cent1.x - cent2.x, 2) + pow(cent1.y - cent2.y, 2));
				if(temp<min_dist){
					min_dist = temp;
				}
			}
		}*/


		Point cent1((boxes[i].br().x+boxes[i].tl().x)/2, (boxes[i].br().y+boxes[i].tl().y)/2);
		Point cent2((boxes[i-1].br().x+boxes[i-1].tl().x)/2, (boxes[i-1].br().y+boxes[i-1].tl().y)/2);
		float temp = sqrt(pow(cent1.x - cent2.x, 2) + pow(cent1.y - cent2.y, 2));
		dist[i] = int(temp);
	}

	int count[int(boxes.size())];

	for(int i=0;i<boxes.size();i++){
		int no = 0;
		for(int j=i+1;j<boxes.size();j++){
			if(abs(dist[i] - dist[j]) < 5)
				no++;
		}
		count[i] = no;
	}

	int max_dist, max_c=0;

	for(int i=0;i<boxes.size();i++){
		if(count[i]>max_c){
			max_c = count[i];
			max_dist = dist[i];
		}
	}

	
	vector<int> nos;

	nos.push_back(0);

	for(int i=1;i<boxes.size();i++){
		if(dist[i]<1.5*max_dist){
			
			nos.push_back(i);
		}
		else{

			final_box.push_back(large_box(boxes, nos));
			nos.clear();
			nos.push_back(i);
		}
	}

	if(nos.size()>0)
		final_box.push_back(large_box(boxes, nos));
}

Mat preprocess1(Mat img)
{
	Mat oriented, dst, dst0, dst1;
	orient(img, oriented);

	dst = oriented.clone();
	dst0 = oriented.clone();
	dst1 = oriented.clone();


	// remove_lines(oriented, dst0);
	// remove_lines(dst0, dst1);
	// remove_lines(dst1, dst);



	return dst;
}

Mat preprocess2(Mat dst)
{

	 Mat img2, img3, filtered, adapt_img , dst0, dst1;
	
	

	// dst = oriented.clone();
	// dst0 = oriented.clone();
	// dst1 = oriented.clone();
	filtered = dst.clone();

	

	remove_lines(oriented, dst0);
	remove_lines(dst0, dst1);
	remove_lines(dst1, dst);

	threshold(dst, adapt_img, 128, 255, THRESH_BINARY|THRESH_OTSU);
	// adaptiveThreshold(dst, adapt_img, 255, BORDER_REPLICATE, THRESH_BINARY, 11, 1);
	// filtered = adapt_img.clone();
	imwrite("adap.jpg", adapt_img);
	kfill(adapt_img, filtered, 4,1);
	imwrite("kfill.jpg", filtered);
	cvtColor(filtered.clone(), img3, COLOR_GRAY2RGB);

	vector<Rect> boundRect;
	vector<Point> lower_mid;
	vector<Point3f> lines;

	// getBoundingBoxes(filtered,  boundRect, lower_mid);
	return filtered;
	// remove_I(filtered, boundRect);
	// vector< vector< Rect> > grp;
	// vector<Point> initial_point;

	

	// improvedBB(boundRect, initial_point, grp);
	// mergeLines(initial_point, grp);

	// Mat bw,r;
	// vector<Mat> final;
	// int s = 128;
	// for (int i = 0; i < boundRect.size(); ++i)
	// {
	// 	// cout << i << endl;
	// 	Mat pad(s, s, CV_8UC1, Scalar(255));
	// 	Rect myROI(boundRect[i].tl().x -2, boundRect[i].tl().y-3,  boundRect[i].width+2, boundRect[i].height+4);
		
	// 	dst0 = dst(myROI);
	// 	threshold(dst0, dst0, 128, 255, THRESH_BINARY|THRESH_OTSU);
		
	// 	resize(dst0, bw , Size(50, 50*float(dst0.rows)/dst0.cols));
	// 	int ht = min(s,bw.rows);
	// 	int wd = min(s,bw.cols);
	// 	// cout << wd << " " <<  ht << endl;
	// 	wd = (s - wd)/2;
	// 	ht = (s - ht)/2;
	// 	cout << wd << " " <<  ht << endl;
	// 	copyMakeBorder(bw, pad, ht, ht, wd, wd, BORDER_CONSTANT, Scalar(255));
	// 	final.push_back(pad);
		

	// }
	// return final;

}

vector< vector<Rect> > preprocess3(Mat filtered)
{
	vector<Rect> boundRect;
	vector<Rect> rejected;
	vector<Point> lower_mid;
	vector<Point3f> lines;
	printf("%s\n", "INSIDE");
	erode(filtered, filtered, Mat(), Point(-1,-1));
	// Mat filtered1 = filtered.clone();
	// kfill(filtered, filtered1, 4,1);
	imwrite("erode.jpg", filtered);
	// dilate(filtered, filtered, Mat(), Point(-1,-1));
	getBoundingBoxes(filtered,  boundRect, lower_mid);
	printf("%s\n","CROSSED" );
	remove_I(filtered, boundRect, rejected);
	vector< vector< Rect> > grp;
	vector<Point> initial_point;

	vector<vector<Rect> > v;

	improvedBB(boundRect, initial_point, grp);
	mergeLines(initial_point, grp);
	// v.push_back(boundRect);
	// v.push_back(rejected);

	grp.push_back(rejected);

	return grp;
}

// int preprocess4(vector<Rect> rejected, Mat img)
// {

// 	// for (int i = 0; i < rejected.size(); ++i)
// 	// {
// 	// 	// Rect roi(rejected[i]);
// 	// 	// Mat crop = img(roi), crop1;

// 	// 	// threshold(crop, crop1, 0 ,255, THRESH_BINARY);
// 	// 	// printf("%s %d %d\n","CHANNELS:", crop.channels(), crop1.channels());
// 	// 	// crop1.copyTo(crop);	
// 	// }
// 	return 1;
	
// }

}
