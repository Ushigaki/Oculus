// main.cpp 
//
//MIT License
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
//THE SOFTWARE.
//rawr fear me
// Oculus Rift : TM & Copyright Oculus VR, Inc. All Rights Reserved
// Wizapply Framework : (C)Wizapply.com

/* -- Include files -------------------------------------------- */
////memo
//299.715	176.112	28.3746
//196.381	200.527	29.0069
//251.026	128.13	159.615

#include "Stdafx.h"
#include "Viewer.h"
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"


// Entrypoint
#include <EntryPoint.h>		//!<Cross platform for common entry point
#ifdef WIN32
#include <ovrvision.h>		//Ovrvision SDK
#include <ovrvision_ar.h>
#else
#include <OvrvisionSDK/ovrvision.h>		//Ovrvision SDK
#include <OvrvisionSDK/ovrvision_ar.h>
#define ShowCursor(x)
#endif

//Oculus SDK
#include "COculusVR.h"
//3DModel
#include "C3DCubeModel.h"

using namespace std;
/* -- Macro definition ------------------------------------------------------- */

//Oculus Rift screen size
#define RIFTSCREEN_WIDTH	(1920)
#define RIFTPSCREEN_HEIGHT	(1080)

//Application screen size
#define APPSCREEN_WIDTH		(1280)
#define APPSCREEN_HEIGHT	(800)

//Camera image size
#define CAM_WIDTH			640
#define CAM_HEIGHT			480

/* -- Global variable definition ----------------------------------------------- */

//Objects
OVR::Ovrvision* g_pOvrvision;
OVR::OvrvisionAR* g_pOvrvisionAR;

//Screen texture
wzTexture g_screen_texture;
wzTexture g_screen_texture2;

COculusVR* g_pOculus;
//C3DCubeModel* g_pCubeModel;
wzMatrix   g_projMat[2];	//left and right projection matrix
wzVector3  g_oculusGap;

//Interocular distance
float eye_interocular = 0.0f;
//Eye scale
float eye_scale = 0.9f;
//Quality
int processer_quality = OV_PSQT_HIGH;
//use AR
bool useOvrvisionAR = false;
bool useCapture = false; //初期キャプチャ
bool init_capture =false;
//flag
int flag = 0;
int poly_num;
int size;
vector <double> before,after;
Eigen::Matrix3d A;
Eigen::Vector3d B;
//double P=0.0;

/* -- Function prototype ------------------------------------------------- */

void UpdateFunc(void);	//!<関数

/*--Definition-------------------------------------------------------------*/
vector <float> vertex;
vector <float> normal;
wzVector2	cursor;		//!<カーソルの位置
int Readstl(const string filepath,vector <float> &normal,vector <float> &vertex);
//Mesh data
wzMesh	m_cube;
//Shader data
wzShaderProg m_shader;
//init translate
float init_t_x=0.0f,init_t_y=0.0f,init_t_z=0.0f;
/*
 *	シェーダプログラムソースGLSL
 */
// バーテックスシェーダ
// 頂点情報とテクスチャＵＶ座標をもらう
static const char* g_c3d_mdsource_vs = {
"attribute vec3 vPos;\n"
"attribute vec3 vNml;\n"
"uniform mat4 u_worldViewProjMatrix;\n"
"varying vec3 v_nml;\n"
"void main(void)\n"
"{\n"
"   v_nml = vNml;\n"
"	gl_Position = u_worldViewProjMatrix * vec4(vPos,1.0);\n"
"}\n"
};

// フラグメントシェーダ
static const char* g_c3d_mdsource_fs = {
"varying vec3 v_nml;\n"
"uniform vec4 u_diffuse;\n"
"uniform vec3 u_lightdiffuse;\n"
"uniform vec3 u_lightdir;\n"
"void main (void)\n"
"{\n"
"   vec4 colors = u_diffuse;"
"	colors *= vec4((u_lightdiffuse * max(dot(v_nml, normalize(u_lightdir)),0) + 0.4) ,1.0);\n"
"   gl_FragColor = colors;\n"
"}\n"
};

cv::CascadeClassifier cascade;
string cascadeName = "./haarcascade_frontalface_alt.xml";//分類器の読み込み
//string cascadeName = "C:/opencv/sources/data/lbpcascades/lbpcascade_frontalface.xml";
cv::Mat init_img;
cv::Mat background;
unsigned char p3[CAM_HEIGHT*CAM_WIDTH*3];
double aaaaa=5.0;
/* -- Functions ------------------------------------------------------------- */
/////////// CLASS ///////////

void C3DCubeModel::DeleteModel()
{
	if(!m_isCreated)
		return;

	wzDeleteMesh(&m_cube);
	wzDeleteShader(&m_shader);
	//wzDeleteTexture(&m_texture); not use

	// Deleted flag
	m_isCreated = false;
}

C3DCubeModel::C3DCubeModel()
{
	// 初期化
	memset(&m_cube,0,sizeof(m_cube));
	memset(&m_shader,0,sizeof(m_shader));
	memset(&m_texture,0,sizeof(m_texture));

	m_isCreated = false;
}
//デストラクタ
C3DCubeModel::~C3DCubeModel()
{
	DeleteModel();
}

/*
 * Initialize application
 */
int Initialize()
{
	//Debug Console ( debug mode )
#ifdef DEBUG
	AllocConsole();
	freopen ( "CONOUT$", "w", stdout ); 
	freopen ( "CONIN$", "r", stdin );
#endif

	//string stl("E:/VisibleHuman_mb.stl");
	string stl("E:/zugaikotu.stl");

	poly_num = Readstl(stl,normal,vertex);

	//Initialize Wizapply library 
	wzInitCreateWizapply("Ovrvision",RIFTSCREEN_WIDTH,RIFTPSCREEN_HEIGHT,WZ_CM_NOVSYNC);//|WZ_CM_FULLSCREEN|WZ_CM_FS_DEVICE

	//Create HMD object　　　　　　　　　　　　　　　
	g_pOculus = new COculusVR();

	/*------------------------------------------------------------------*/
	// Library setting
	wzSetClearColor(0.0f,0.0f,0.0f,1);
	wzSetSpriteScSize(APPSCREEN_WIDTH, APPSCREEN_HEIGHT);	// Sprite setting
	wzSetCursorScSize(APPSCREEN_WIDTH, APPSCREEN_HEIGHT);	// Screen cursor setting
	//Create ovrvision object
	g_pOvrvision = new OVR::Ovrvision();
	if(g_pOculus->GetHMDType() == ovrHmd_DK2) {
		//Rift DK2
		g_pOvrvision->Open(0,OVR::OV_CAMVGA_FULL);	//Open
	} else {
		//Rift DK1
		g_pOvrvision->Open(0,OVR::OV_CAMVGA_FULL,OV_HMD_OCULUS_DK1);	//Open
	}
	//Create ovrvision ar object
	g_pOvrvisionAR = new OVR::OvrvisionAR(0.15f,CAM_WIDTH,CAM_HEIGHT,g_pOvrvision->GetFocalPoint());
	cout<<"FocalPoint is"<<endl;
	cout<<g_pOvrvision->GetFocalPoint()<<endl;
	if(g_pOvrvision->GetFocalPoint() == 0)
		exit(1);
	//Create Projection
	wzMatrixPerspectiveFovLH(&g_projMat[0], g_pOculus->GetHMDFov(0), g_pOculus->GetHMDAspect(), 0.1f, 1000.0f);
	wzMatrixPerspectiveFovLH(&g_projMat[1], g_pOculus->GetHMDFov(1), g_pOculus->GetHMDAspect(), 0.1f, 1000.0f);

	//OculusRightGap
	g_oculusGap.x = g_pOvrvision->GetOculusRightGap(0) * -0.01f;
	g_oculusGap.y = g_pOvrvision->GetOculusRightGap(1) * 0.01f;
	g_oculusGap.z = g_pOvrvision->GetOculusRightGap(2) * 0.01f;

	//Hide cursor
	ShowCursor(TRUE);

	//Create texture
	wzCreateTextureBuffer(&g_screen_texture, CAM_WIDTH, CAM_HEIGHT, WZ_FORMATTYPE_RGB);
	wzCreateTextureBuffer(&g_screen_texture2, CAM_WIDTH, CAM_HEIGHT, WZ_FORMATTYPE_RGB);

	/*------------------------------------------------------------------*/
	//Polaris and GL init
	
	SampleViewer sampleviewer;
	//sampleviewer.INITIALIZEDPOLARIS();
	//sampleviewer.INITIOPENGLHOOKS();
	//Set update function 
	wzSetUpdateThread(60,UpdateFunc);	// UpdateFun
	//printf("顎のサイズを入力してください\n");
	//scanf("%d",&size);
	if(!cascade.load(cascadeName)){
			cout << "Loading Error!!" <<endl;
			exit(-1);
			}
	return 0;
}

/*
 * Application exit
 */
int Terminate()
{
	//Delete object
	delete g_pOvrvision;
	delete g_pOvrvisionAR;
	delete g_pOculus;
//	delete g_pCubeModel;

	wzDeleteTexture(&g_screen_texture);
	wzDeleteTexture(&g_screen_texture2);

	ShowCursor(TRUE);

	/*------------------------------------------------------------------*/

	//Shutdown Wizapply library
	wzExitWizapply();

	return 0;
}




void OculusEndFrame()
{
	if(!g_pOculus->isReady()) g_pOculus->EndFrameTiming();
}

//stlファイル読み込み
vector <string> ParseStl(const string str)
{
	vector<string> result;
	string buf;
	string::const_iterator bit, it;
	bool readchar = false;
	bit = str.begin();
	while(bit != str.end()){
		if(((*bit == 0x20)||(*bit =='\t')) && !readchar) {
			readchar = false;
			bit++;
			continue;
		}
		if(((*bit != 0x20)||(*bit !='\t')) && !readchar) {
			readchar = true;
			it = bit;
			bit++;
			continue;
		}
		if(((*bit == 0x20)||(*bit =='\t')) && readchar) {
			readchar = false;
			buf.assign("");
			buf.append(it,bit);
			result.push_back(buf);
			bit++;
			continue;
		}
		bit++;
	}
	
	return result;
}

//stl読み込み
int Readstl(const string filepath,vector <float> &normal,vector <float> &vertex){
	vector<string> pstr;
	ifstream fs(filepath.c_str(), ios::in);
	int num=0;
	if(!fs) {
		cout << "The file can't open." << endl;
		exit(0);
	}
	string str;
	string::iterator bit,it1,it2;
	
	while(!fs.eof()) {
		getline(fs, str);
		
		str.append(" ");
		pstr = ParseStl(str);
		 
		if(pstr.size() > 0){
			if(pstr.at(0) == string("facet")) {
				if(pstr.at(1) == string("normal")){
					for(int i=0;i<3;i++){
					normal.push_back( atof(pstr.at(2).c_str()));
					normal.push_back( atof(pstr.at(3).c_str()));
					normal.push_back( atof(pstr.at(4).c_str()));
					}

				}
			}
			if(pstr.at(0) == string("vertex")) {
				if(num!=0){
				vertex.push_back( atof(pstr.at(1).c_str())-vertex[0]);
				vertex.push_back( atof(pstr.at(2).c_str())-vertex[1]);
				vertex.push_back( atof(pstr.at(3).c_str())-vertex[2]);
				}
				else{
				/*vertex.push_back( atof(pstr.at(1).c_str()));
				vertex.push_back( atof(pstr.at(2).c_str()));
				vertex.push_back( atof(pstr.at(3).c_str()));*/
				vertex.push_back(0.0);
				vertex.push_back(0.0);
				vertex.push_back(0.0);
				}
				num++;
			}
		}
	}
	
	fs.close();
	return num;
}

float point(double x,double y,double x2,double y2,double x3,double y3){

	cout<<"start"<<endl<<x<<endl<<y<<endl<<x2<<endl<<y2<<endl<<x3<<endl<<y3<<endl<<"End"<<endl;
	
	before.push_back(x);//left x
	before.push_back(y);//left y
	before.push_back(-70.0);//left z
	before.push_back(x2);//right x
	before.push_back(y2);//right y
	before.push_back(-70.0);//right z
	before.push_back(x3);//center x
	before.push_back(y3);//center y
	before.push_back(-50.0);//center z

	//mandible bone points
	after.push_back(299.715);//mondible left
	after.push_back(176.112);
	after.push_back(28.3746);
	after.push_back(196.381);//right
	after.push_back(200.527);
	after.push_back(29.0069);
	after.push_back(251.026);//center
	after.push_back(128.13);
	after.push_back(159.615);


	float size[6]={0.0,0.0,0.0,0.0,0.0,0.0}, ave=0.0, sum=0.0;
	size[0] = fabs((before[3] - before[0]) / (after[3] - after[0]));//x
	size[1] = fabs((before[5] - before[2]) / (after[4] - after[1]));//y
	size[2] = fabs((before[4] - before[1]) / (after[5] - after[2]));//z
	size[3] = fabs((before[6] - before[0]) / (after[6] - after[0]));//x
	size[4] = fabs((before[8] - before[2]) / (after[7] - after[1]));//y
	size[5] = fabs((before[7] - before[1]) / (after[8] - after[2]));//z
	
	for(int i=0;i<6;i++){
		sum += size[i];
	}


	ave = sum / 6.0;


	for(int i=0;i<9;i++){
		after[i] = after[i] * ave;
	}
	return ave;
}

//平行移動+回転
void affine(){
	Eigen::Vector3d x1,x2,x3,y,y2,y3,q1,p1;
	Eigen::Matrix3d X,Y;

	for(int i= 0; i < 3; i++){
		q1(i) = before[i];//アフィン変換による像
		y(i)  = before[i+3] - before[i];
		y2(i) = before[i+6] - before[i];
		p1(i) = after[i];//変換前の像
		x1(i) = after[i+3] - after[i];
		x2(i) = after[i+6] - after[i];
	}

	y3 = y.cross(y2);
	x3 = x1.cross(x2);

	X.col(0) = x1;
	X.col(1) = x2;
	X.col(2) = x3;
	Y.col(0) = y;
	Y.col(1) = y2;
	Y.col(2) = y3;

	A = Y * X.inverse();
	B = q1 - (A * p1);
}




void DrawModel(wzMatrix* pProj, wzVector3* gap,float &p_q0,float &p_qx,float &p_qy,float &p_qz,float &t_x,float &t_y,float &t_z,float &t_x2,float &t_y2,float &t_z2)
{

	wzMatrix wvpMatans;
	wzVector4 modelColor = VEC4_INIT(1.0f,1.0f,1.0f,1.0f);
	wzVector3 lightColor = VEC3_INIT(1.0f,1.0f,1.0f);
	wzVector3 lightDirection = VEC3_INIT(0.6f,0.7f,1.0f);

	// Matrix multiply
	wzMatrixIdentity(&wvpMatans);
	wzMatrixMultiply(&wvpMatans, pProj, &wvpMatans);
	
	if(gap != NULL) {
		wzMatrix drawcalc,rotate,translation;
		wzQuaternion quat;
		quat.w = p_q0;
		quat.x = p_qx;
		quat.y = p_qy;
		quat.z = p_qz;
		wzQuaternionToMatrix(&rotate,&quat);
		wzMatrixTranslation(&drawcalc,gap->x, gap->y, gap->z);
		wzMatrixTranslation(&translation,t_x,t_y,t_z);
		wzMatrixMultiply(&wvpMatans, &drawcalc, &wvpMatans);
		wzMatrixMultiply(&wvpMatans, &translation, &wvpMatans);
		wzMatrixMultiply(&wvpMatans, &rotate, &wvpMatans);
		}

	

	// Shader
	wzUseShader(&m_shader);
	wzUniformMatrix("u_worldViewProjMatrix",&wvpMatans);
	// Color
	wzUniformVector4("u_diffuse",&modelColor);
	wzUniformVector3("u_lightdiffuse",&lightColor);
	wzUniformVector3("u_lightdir",&lightDirection);
	// Drawing
	wzDrawMesh(&m_cube);
}


void DrawLoop(void)
{
	float p_q0=1.0f, p_qx=0.0f, p_qy=0.0f, p_qz=0.0f, t_x=0.0f, t_y=0.0f, t_z=0.0f, t_x2=0.0f, t_y2=0.0f, t_z2=0.0f;
	
	float polaris_r[16],polaris_r2[16];
	for(int i=0 ;i<=15 ;i++){
		polaris_r[i] = 0.0f;
		polaris_r2[i] = 0.0f;
	}
	polaris_r[0] = 1.0f;
	polaris_r[5] = 1.0f;
	polaris_r[10] = 1.0f;
	polaris_r[15] = 1.0f;
	polaris_r2[0] = 1.0f;
	polaris_r2[5] = 1.0f;
	polaris_r2[10] = 1.0f;
	polaris_r2[15] = 1.0f;
	
	float model_scale;
	SampleViewer sampleviewer;
	
	wzSetDepthTest(TRUE);		//Depth off
	wzSetCullFace(WZ_CLF_NONE);	//Culling off
	wzVector2 half_pos = {APPSCREEN_WIDTH/2/2,APPSCREEN_HEIGHT/2};
	if(!g_pOculus->isReady())
	{
		// Get ovrvision image
		g_pOvrvision->PreStoreCamData();	//renderer(両目の画像取得)
		unsigned char* p = g_pOvrvision->GetCamImage(OVR::OV_CAMEYE_LEFT, (OvPSQuality)processer_quality);//左目の画像取得
		unsigned char* p2 = g_pOvrvision->GetCamImage(OVR::OV_CAMEYE_RIGHT, (OvPSQuality)processer_quality);//右目の画像取得
		if(init_capture){
			//p3=g_pOvrvision->GetCamImage(OVR::OV_CAMEYE_LEFT, (OvPSQuality)processer_quality);//左目の画像取得
			//g_pOvrvision->GetCamImage(p3,OVR::OV_CAMEYE_LEFT,(OvPSQuality)processer_quality);//左目の画像取得
			memmove(p3,p,CAM_HEIGHT*CAM_WIDTH*3);
			background = cv::Mat(CAM_HEIGHT,CAM_WIDTH,CV_8UC3,p3);
			init_capture = false;
		}

		int scale = 4;
		cv::Mat img(CAM_HEIGHT,CAM_WIDTH,CV_8UC3,p);
		//cv::Mat img = cv::imread("./lena.png");
		

		
		//AR
		if(useOvrvisionAR) {
			g_pOvrvisionAR->SetImageRGB(p);	//(左目の画像を基準にしてARを行う)
			g_pOvrvisionAR->Render();//ARマーカを検知
		}

		//render
		g_pOculus->BeginDrawRender();

		// Screen clear
		wzClear();

		// Left eye
		g_pOculus->SetViewport(0);
		wzSetSpriteScSize(APPSCREEN_WIDTH/2, APPSCREEN_HEIGHT);

		// Shader vertex format
		wzVertexElements ve_var[] = {
			{WZVETYPE_FLOAT3,"vPos"},
			{WZVETYPE_FLOAT3,"vNml"},
			WZVE_TMT()
		};

		// Create shader.
		if(wzCreateShader_GLSL(&m_shader,
			g_c3d_mdsource_vs,g_c3d_mdsource_fs,ve_var)){
				cout<<"Shader Error!!"<<endl;
				exit(EXIT_FAILURE);
		}

		//マウス座標取得
		//while(flag == 0){
		/*	if(WM_LBUTTONDOWN){
				if(wzGetCursorSrPos(0,&cursor.x,&cursor.y))
					cout<<cursor.x<<endl;
			}*/
		//}

	
		if(useOvrvisionAR){
			//cv::Mat img = cv::imread("./lena.png");
			//cv::cvtColor(img,img,CV_RGB2BGR);
			cv::Mat gray, smallImg(cv::saturate_cast<int>(img.rows/scale),cv::saturate_cast<int>(img.cols/scale),CV_8UC1);
			cv::cvtColor(img,gray,CV_BGR2GRAY);//グレースケール
			cv::resize(gray,smallImg,smallImg.size(),0,0,cv::INTER_LINEAR);//縮小
			cv::equalizeHist(smallImg,smallImg);
			vector<cv::Rect> faces;
			
			//顔探索
			cascade.detectMultiScale(smallImg,faces,
									 1.1,8,
									 CV_HAAR_SCALE_IMAGE,
									 cv::Size(10,10));

			vector<cv::Rect>::const_iterator r = faces.begin();
			////11/30 顔領域に輝度の差が生じている　原因は照明光にあると思われる　修正すること　
			for(; r != faces.end(); ++r){					
					if(!background.empty()){
						useCapture = true;
						for(int y = r->y*scale;y<(r->height+r->y)*scale;y++){
							for(int x = (int)r->x*scale; x< (int)(r->width+r->x)*scale;x++){
								for(int c = 0;c<img.channels();c++){
									img.data[x*img.elemSize()+y*img.step+c] = background.data[x*img.elemSize()+y*img.step+c];
									//img.data[x*img.elemSize()+y*img.step+c] = init_img.data[x*init_img.elemSize()+y*init_img.step+c];
								}
							}
						}		
					}
				if(useCapture){
					model_scale = point(r->x*scale,r->y*scale,
										(r->width+r->x)*scale,r->y*scale,
										((r->width+r->x)*scale+r->x*scale)/2,(r->y+r->height)*scale);
					affine();
					}
				 /*cv::Point center;
				 int radius;
				 center.x = cv::saturate_cast<int>((r->x + r->width*0.5)*scale);
				 center.y = cv::saturate_cast<int>((r->y + r->height*0.5)*scale);
				 radius = cv::saturate_cast<int>((r->width + r->height)*0.25*scale);
				 cv::circle(img, center, radius, cv::Scalar(80,80,255), 3, 8 );
				 
				 wzSetSpriteColor(1.0f, 1.0f, 1.0f, 1.0f);
				 wzSetSpriteRotate(0.0f);
				 wzFontSize(12);
				 wzPrintf(20, 120, "Face detection");
				 cout<<r->width<<endl;*/
				}
			
			/*faces.clear();
			gray.release();
			smallImg.release();
			img.release();*/

		}
		after.clear();
		before.clear();


		if(useCapture){
		Eigen::Vector3d convert_p;
		for(int i=0;i<vertex.size();i+=3){
			convert_p(0)= vertex[i]  *model_scale;
			convert_p(1)= vertex[i+1]*model_scale;
			convert_p(2)= vertex[i+2]*model_scale;
			convert_p   = A * convert_p + B;
			vertex[i]   = convert_p(0);
			vertex[i+1] = convert_p(1);
			vertex[i+2] = convert_p(2);
			}
		}



		const float* vertex_pointer1[] = {&vertex[0],&normal[0]};
		if(wzCreateMesh(&m_cube, (void**)&vertex_pointer1, ve_var, NULL, vertex.size()/3, vertex.size()/9)) {
		exit(1);
		}

		//wzChangeDrawMode(&m_cube,WZ_MESH_DF_TRIANGLELIST);
		if(p != NULL) {
			//sampleviewer.GetTrackingData(p_q0,p_qx,p_qy,p_qz,t_x,t_y,t_z,t_x2,t_y2,t_z2);//p:クォータニオン,t:患者の座標,t_1:oculusの座標
			if(flag==0){
				init_t_x = t_x;
				init_t_y = t_y;
				init_t_z = t_z;
				flag++;
			}
			t_x -= init_t_x;
			t_y = -1*(t_y - init_t_y);
			t_z = -1*(t_z - init_t_z);
			//uchar tmp[640][480];
			//unsigned char* p3;
			//memcpy(p,img.data,sizeof(char)*CAM_WIDTH*CAM_HEIGHT);
			
			wzChangeTextureBuffer(&g_screen_texture, 0, 0, CAM_WIDTH, CAM_HEIGHT, WZ_FORMATTYPE_RGB, (char*)img.data, 0);
			wzSetSpritePosition(eye_interocular + half_pos.x, half_pos.y, 0.0f);
			wzSetSpriteColor(1.0f, 1.0f, 1.0f, 1.0f);
			wzSetSpriteTexCoord(0.0f, 0.0f, 1.0f, 1.0f);
			wzSetSpriteSize((float)CAM_WIDTH * eye_scale, (float)CAM_HEIGHT * eye_scale);
			wzSetSpriteTexture(&g_screen_texture);
			wzSpriteDraw();	//Draw
			
			if(useCapture)
			DrawModel(&g_projMat[0], &g_oculusGap,p_q0,p_qx,p_qy,p_qz,t_x,t_y,t_z,t_x2,t_y2,t_z2);//テスト用に削除可
			
		}
		//やりたいのはここで顔検出からの輪郭検出からの顔の向き検出からの骨の回転＆重畳表示(不必要??)
		//代わりに卒論のプログラムとの合体(Xtionの代わりがOculusのカメラ)
		//やることは、Oculusと患者にマーカをつけ、その位置情報から顎の位置推定を行い、下顎骨を重畳表示する
		//Diminished Realityの実装
	 	
		//3DCube left Renderer
		//if(useOvrvisionAR) {
		//	OVR::OvMarkerData* dt = g_pOvrvisionAR->GetMarkerData();
		//	for(int i=0; i < g_pOvrvisionAR->GetMarkerDataSize(); i++)
		//		if(dt[i].id==64) DrawARModel(&dt[i], &g_projMat[0],NULL);	//id=64
		//}
		
		// Right eye
		g_pOculus->SetViewport(1);
		wzSetSpriteScSize(APPSCREEN_WIDTH/2, APPSCREEN_HEIGHT);
		if(p2 != NULL) {
			wzChangeTextureBuffer(&g_screen_texture2, 0, 0, CAM_WIDTH, CAM_HEIGHT, WZ_FORMATTYPE_RGB, (char*)p, 0);
			wzSetSpritePosition((-eye_interocular)+half_pos.x, half_pos.y, 0.0f);
			wzSetSpriteColor(1.0f, 1.0f, 1.0f, 1.0f);
			wzSetSpriteTexCoord(0.0f, 0.0f, 1.0f, 1.0f);
			wzSetSpriteSize((float)CAM_WIDTH * eye_scale, (float)CAM_HEIGHT * eye_scale);
			wzSetSpriteTexture(&g_screen_texture2);
			wzSpriteDraw();	//Draw
			//DrawModel( &g_projMat[1], &g_oculusGap,p_q0,p_qx,p_qy,p_qz,t_x,t_y,t_z,t_x2,t_y2,t_z2);//テスト用
		}

		//3DCube right Renderer
		//if(useOvrvisionAR) {
			//OVR::OvMarkerData* dt = g_pOvrvisionAR->GetMarkerData();
			//for(int i=0; i < g_pOvrvisionAR->GetMarkerDataSize(); i++)
			//	if(dt[i].id==64) g_pCubeModel->DrawARModel(&dt[i], &g_projMat[1], &g_oculusGap);
		//}

		//EndRender
		g_pOculus->EndDrawRender();

		//ScreenDraw
		g_pOculus->DrawScreen();
	} else {
		// Screen clear
		wzClear();
		
	}
	useCapture = false;
	/*------------------------------------------------------------------*/

	wzSetDepthTest(FALSE);		//Depth off
	//Full Draw
	wzSetViewport( 0, 0, 0, 0 );
	wzSetSpriteScSize(APPSCREEN_WIDTH, APPSCREEN_HEIGHT);

	// Debug infomation
	wzSetSpriteColor(1.0f, 1.0f, 1.0f, 1.0f);
	wzSetSpriteRotate(0.0f);
	wzFontSize(12);

	wzPrintf(20, 30, "Ovrvision DemoApp v2");
	wzPrintf(20, 60, "Update:%.2f, Draw:%.2f", wzGetUpdateFPS(),wzGetDrawFPS());
	wzPrintf(20, 75, "EYE DISTANCE :%.2f , EYE SCALE :%.2f PS_QT :%d", eye_interocular, eye_scale, processer_quality);
	wzPrintf(20, 735, "EYE DISTANCE SETTING: Left Key & Right Key");
	wzPrintf(20, 750, "PROCESS QUALITY: 1,2,3,4 Key TOGGLE AR MODE : A Key");


	//AR infomation
	if(useOvrvisionAR) {
		wzPrintf(20, 100, "Ovrvision AR mode!");

		OVR::OvMarkerData* dt = g_pOvrvisionAR->GetMarkerData();
		for(int i=0; i < g_pOvrvisionAR->GetMarkerDataSize(); i++)
		{
			wzPrintf(20, 115+(i*20), "ARMarker:%d, T:%.2f,%.2f,%.2f Q:%.2f%.2f,%.2f,%.2f", dt[i].id, 
				dt[i].translate.x, dt[i].translate.y, dt[i].translate.z,
				dt[i].quaternion.x, dt[i].quaternion.y, dt[i].quaternion.z, dt[i].quaternion.w);
		}
	}

	//Error infomation
	if(g_pOvrvision->GetCameraStatus() == OVR::OV_CAMCLOSED) {
		wzSetSpriteColor(1.0f, 0.0f, 0.0f, 1.0f);
		wzPrintf(20, 120, "[ERROR]Ovrvision not found.");
	}
	if(g_pOculus->isReady()) {
		wzSetSpriteColor(1.0f, 0.0f, 0.0f, 1.0f);
		wzPrintf(20, 120, "[ERROR]Oculus Rift not found.");
	}

}

/*
 * Update Thread
 * This function is called every fixed interval of 60fps.
 */
void UpdateFunc(void)
{
	//Control program
	if(wzGetKeyStateTrigger(WZ_KEY_UP))
		eye_scale -= 0.01f;
	if(wzGetKeyStateTrigger(WZ_KEY_DOWN))
		eye_scale += 0.01f;
	if(wzGetKeyStateTrigger(WZ_KEY_LEFT))
		eye_interocular -= 3.0f;
	if(wzGetKeyStateTrigger(WZ_KEY_RIGHT))
		eye_interocular += 3.0f;

	if(wzGetKeyStateTrigger(WZ_KEY_1))
		processer_quality = OV_PSQT_NONE;
	if(wzGetKeyStateTrigger(WZ_KEY_2))
		processer_quality = OV_PSQT_LOW;
	if(wzGetKeyStateTrigger(WZ_KEY_3))
		processer_quality = OV_PSQT_HIGH;
	if(wzGetKeyStateTrigger(WZ_KEY_4))
		processer_quality = OV_PSQT_REFSET;
	if(wzGetKeyStateTrigger(WZ_KEY_A))
		useOvrvisionAR ^= true;
	if(wzGetKeyStateTrigger(WZ_KEY_Q))
		exit(0);
	if(wzGetKeyStateTrigger(WZ_KEY_SPACE))
		init_capture = true;
}

	
	
	
	


//EOF
