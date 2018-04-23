#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include <iostream>
#include <stdio.h>
#include <math.h>


using namespace std;
using namespace cv;

//Variaveis:
int escolha=1;
Mat dest;
Mat img;

string face_cascade_name = "haarcascade_frontalface_alt.xml";
CascadeClassifier face_cascade;
string window_name = "Capture - Face detection";
int filenumber; // Number of file to be saved
int i;
string filename;
string nomeArquivo;
string nomeArquivoFinal;
double lm = 0.5+1/100.0;

Size size(400,400);//the dst image size,e.g.100x100
Mat dst;//dst image
Mat src;//src image
Mat newtam;

Mat GaborCurvo2(int ks, double sig, double th, double lm, double ps, double curva){
    int c=curva;
    int hks = (ks-1)/2;
    double theta = th*CV_PI/180;
    double psi = ps*CV_PI/180;
    double del = 2.0/(ks-1);
    double lmbd = lm;
    double sigma = sig/ks;
    double x_theta;
    double y_theta;
    Mat kernel(ks,ks, CV_32F);
    for (int y=-hks; y<=hks; y++)
    {
        for (int x=-hks; x<=hks; x++)
        {
            y_theta = -x*del*sin(theta)+y*del*cos(theta);
            x_theta = x*del*cos(theta)+y*del*sin(theta)+(c*(pow(y_theta,2)));

            kernel.at<float>(hks+y,hks+x) = (float)exp(-0.5*(pow(x_theta,2)+pow(y_theta,2))/pow(sigma,2))* cos(2*CV_PI*x_theta/lmbd + psi);
        }
    }
    return kernel;
}

Mat filterApplication(Mat imagem){
    Mat i1 = imagem;
    Mat i2;
    cvtColor(i1, i2, CV_BGR2GRAY);
    Mat i3;
    i2.convertTo(i3, CV_32F, 1.0/255, 0);
    Mat kernel = GaborCurvo2(21, 2, 180, lm, 90,2);
    Mat imFinal;
    filter2D(i3, imFinal, CV_32F, kernel);

    return imFinal;
}

Mat faceDetector(Mat photo){
    std::vector<Rect> faces;
    Mat photo_gray;
    Mat crop;
    Mat res;
    Mat gray;
    string text;
    stringstream sstm;

    cvtColor(photo, photo_gray, COLOR_BGR2GRAY);
    equalizeHist(photo_gray, photo_gray);

// Detect faces
    face_cascade.detectMultiScale(photo_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

// Set Region of Interest
    Rect roi_b;
    Rect roi_c;

    size_t ic = 0; // ic is index of current element
    int ac = 0; // ac is area of current element

    size_t ib = 0; // ib is index of biggest element
    int ab = 0; // ab is area of biggest element

    for (ic = 0; ic < faces.size(); ic++) // Iterate through all current elements (detected faces)

    {
        roi_c.x = faces[ic].x;
        roi_c.y = faces[ic].y;
        roi_c.width = (faces[ic].width);
        roi_c.height = (faces[ic].height);

        ac = roi_c.width * roi_c.height; // Get the area of current element (detected face)

        roi_b.x = faces[ib].x;
        roi_b.y = faces[ib].y;
        roi_b.width = (faces[ib].width);
        roi_b.height = (faces[ib].height);

        ab = roi_b.width * roi_b.height; // Get the area of biggest element, at beginning it is same as "current" element

        if (ac > ab)
        {
            ib = ic;
            roi_b.x = faces[ib].x;
            roi_b.y = faces[ib].y;
            roi_b.width = (faces[ib].width);
            roi_b.height = (faces[ib].height);
        }


    }
        crop = photo(roi_b);
        resize(crop, res, Size(400, 400), 0, 0, INTER_LINEAR); // This will be needed later while saving images
        cvtColor(crop, gray, CV_BGR2GRAY); // Convert cropped image to Grayscale

        Mat aux;
        resize(gray,aux,size);

        return aux;
}

int main(int argc, char** argv){

    while(escolha!=0){
        cout<< "1" <<endl;
        cout<< "2 - DETECTAR FACES"<<endl;
        cout<< "3 - APLICAR FILTRO NAS FACES"<<endl;
        cin>>escolha;
        if(escolha == 1){
            if (!face_cascade.load(face_cascade_name))
            {
                printf("--(!)Error loading\n");
                return (-1);
            };
            i=1;

                nomeArquivo = "";
                stringstream identicador;
                identicador << i;
                nomeArquivo = "pessoa.1."+identicador.str()+".jpg";
                cout<<nomeArquivo<<endl;
                Mat readImage = imread(nomeArquivo,1);

                Mat detectedFace = faceDetector(readImage);

                imshow("RESULTADO PARCIAL",detectedFace);
                waitKey();

                Mat grayDetectedFace;

                cvtColor(detectedFace, grayDetectedFace, CV_BGR2GRAY);

                Mat imageWithFitler = filterApplication(grayDetectedFace);

                imshow("RESULTADO FINAL",imageWithFitler);
                waitKey();


            return main(0,0);

        }else if(escolha == 2){

            if (!face_cascade.load(face_cascade_name)){
                printf("--(!)Error loading\n");
                return (-1);
            };

            for(i=1;i<=2;i++){

                nomeArquivo = "";
                stringstream identicador;
                identicador << i;
                nomeArquivo = "pessoa.1."+identicador.str()+".jpg";
                cout<<nomeArquivo+" OK"<<endl;
                nomeArquivoFinal = "face"+nomeArquivo;

                Mat readImage = imread(nomeArquivo,1);

                Mat detectedFace = faceDetector(readImage);

                imshow("FACE ENCONTRADA",detectedFace);
                imwrite(nomeArquivoFinal,detectedFace);
                waitKey();
            }

            return main(0,0);

        }else if(escolha == 3){

            if (!face_cascade.load(face_cascade_name)){
                printf("--(!)Error loading\n");
                return (-1);
            };
            for(i=1;i<=2;i++){
                nomeArquivo = "";
                stringstream identicador;
                identicador << i;
                nomeArquivo = "facepessoa.1."+identicador.str()+".jpg";
                cout<<nomeArquivo+" OK"<<endl;

                Mat imagem;
                imagem = imread("img2.jpg",1);

                imshow("FACE",imagem);

                Mat imgFilter = filterApplication(imagem);

                nomeArquivoFinal = "filtro"+nomeArquivo;

                imshow(nomeArquivoFinal,imgFilter);

                imwrite("david.jpg",imgFilter);

                waitKey();
            }

            return main(0,0);

        }

    }//END WHILE
}//END MAIN

