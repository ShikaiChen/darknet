#include "network.h"
#include "detection_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "demo.h"
#include "image.h"
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <stdio.h>
#include <sys/stat.h>
#include<fcntl.h> 
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#endif

char *voc_names[] = {"aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"};
image voc_labels[20];


const char _Base[]={"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="};

static union
{
    struct  
    {
        unsigned long a:6;
        unsigned long b:6;
        unsigned long c:6;
        unsigned long d:6;
    }Sdata;
    unsigned char c[3];
}Udata;
char * Encbase64(char * orgdata,unsigned long orglen,unsigned long *newlen)  
{  
    char *p=NULL,*ret=NULL;  
    int tlen=0;  
    if (orgdata==NULL|| orglen==0)  
        return NULL ;  
    tlen=orglen/3;  
    if(tlen%3!=0) tlen++;  
    tlen=tlen*4;  
    *newlen=tlen;  
    if ((ret=(char *)malloc(tlen+1))==NULL)  
        return NULL;  
    memset(ret,0,tlen+1);  
    p=orgdata;tlen=orglen;  
  
    int i=0,j=0;  
    while(tlen>0)  
    {  
        Udata.c[0]=Udata.c[1]=Udata.c[2]=0;  
        for (i=0;i<3;i++)  
        {  
            if (tlen<1) break;  
            Udata.c[i]=(char)*p;  
            tlen--;  
            p++;  
        }  
        if (i==0) break;  
        switch (i)  
        {  
            case 1:  
                /*ret[j++]=_Base[Udata.Sdata.d];  
                ret[j++]=_Base[Udata.Sdata.c];  
                ret[j++]=_Base[64];  
                ret[j++]=_Base[64];*/  
                ret[j++]=_Base[Udata.c[0]>>2];  
                ret[j++]=_Base[((Udata.c[0]&0x03)<<4)|((Udata.c[1]&0xf0)>>4)];  
                ret[j++]=_Base[64];  
                ret[j++]=_Base[64];  
                break;  
            case 2:  
                /*ret[j++]=_Base[Udata.Sdata.d];  
                ret[j++]=_Base[Udata.Sdata.c];  
                ret[j++]=_Base[Udata.Sdata.b];  
                ret[j++]=_Base[64];*/  
                ret[j++]=_Base[Udata.c[0]>>2];  
                ret[j++]=_Base[((Udata.c[0]&0x03)<<4)|((Udata.c[1]&0xf0)>>4)];  
                ret[j++]=_Base[((Udata.c[1]&0x0f)<<2)|((Udata.c[2]&0xc0)>>6)];  
                ret[j++]=_Base[64];  
                break;  
            case 3:  
                /*ret[j++]=_Base[Udata.Sdata.d];  
                ret[j++]=_Base[Udata.Sdata.c];  
                ret[j++]=_Base[Udata.Sdata.b];  
                ret[j++]=_Base[Udata.Sdata.a];*/  
                ret[j++]=_Base[Udata.c[0]>>2];  
                ret[j++]=_Base[((Udata.c[0]&0x03)<<4)|((Udata.c[1]&0xf0)>>4)];  
                ret[j++]=_Base[((Udata.c[1]&0x0f)<<2)|((Udata.c[2]&0xc0)>>6)];  
                ret[j++]=_Base[Udata.c[2]&0x3f];  
                break;  
            default:  
                break;  
        }  
    }  
    ret[j]='\0';  
    return ret;  
} 
char * Decbase64(char * orgdata,unsigned long orglen, unsigned long *dstlen)
{
    char *p,*ret;
    int len;
    char ch[4]={0};
    char *pos[4];
    int  offset[4];
    if (orgdata==NULL || orglen==0)
    {
        return NULL;
    }
    len=orglen*3/4;
    if ((ret=(char *)malloc(len+1))==NULL)
    {
        return NULL;
    }
    p=orgdata;
    len=orglen;
    int j=0;
    
    while(len>0)
    {
        int i=0;
        while(i<4)
        {
            if (len>0)
            {
                ch[i]=*p;
                p++;
                len--;
                if ((pos[i]=(char *)strchr(_Base,ch[i]))==NULL)
                {
                    return NULL;
                }
                offset[i]=pos[i]-_Base;
                
            }
            i++;
        }
        if (ch[0]=='='||ch[1]=='='||(ch[2]=='='&&ch[3]!='='))
        {
            return NULL;
        }
        ret[j++]=(unsigned char)(offset[0]<<2|offset[1]>>4);
        ret[j++]=offset[2]==64?'\0':(unsigned char)(offset[1]<<4|offset[2]>>2);
        ret[j++]=offset[3]==64?'\0':(unsigned char)((offset[2]<<6&0xc0)|offset[3]);
    }
    ret[j]='\0';
    *dstlen = j;
    return ret;
}

void train_yolo(char *cfgfile, char *weightfile)
{
    char *train_images = "/data/voc/train.txt";
    char *backup_directory = "/home/pjreddie/backup/";
    srand(time(0));
    data_seed = time(0);
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    float avg_loss = -1;
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    int imgs = net.batch*net.subdivisions;
    int i = *net.seen/imgs;
    data train, buffer;


    layer l = net.layers[net.n - 1];

    int side = l.side;
    int classes = l.classes;
    float jitter = l.jitter;

    list *plist = get_paths(train_images);
    //int N = plist->size;
    char **paths = (char **)list_to_array(plist);

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.classes = classes;
    args.jitter = jitter;
    args.num_boxes = side;
    args.d = &buffer;
    args.type = REGION_DATA;

    pthread_t load_thread = load_data_in_thread(args);
    clock_t time;
    //while(i*imgs < N*120){
    while(get_current_batch(net) < net.max_batches){
        i += 1;
        time=clock();
        pthread_join(load_thread, 0);
        train = buffer;
        load_thread = load_data_in_thread(args);

        printf("Loaded: %lf seconds\n", sec(clock()-time));

        time=clock();
        float loss = train_network(net, train);
        if (avg_loss < 0) avg_loss = loss;
        avg_loss = avg_loss*.9 + loss*.1;

        printf("%d: %f, %f avg, %f rate, %lf seconds, %d images\n", i, loss, avg_loss, get_current_rate(net), sec(clock()-time), i*imgs);
        if(i%1000==0 || (i < 1000 && i%100 == 0)){
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
            save_weights(net, buff);
        }
        free_data(train);
    }
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
}

void convert_detections(float *predictions, int classes, int num, int square, int side, int w, int h, float thresh, float **probs, box *boxes, int only_objectness)
{
    int i,j,n;
    //int per_cell = 5*num+classes;
    for (i = 0; i < side*side; ++i){
        int row = i / side;
        int col = i % side;
        for(n = 0; n < num; ++n){
            int index = i*num + n;
            int p_index = side*side*classes + i*num + n;
            float scale = predictions[p_index];
            int box_index = side*side*(classes + num) + (i*num + n)*4;
            boxes[index].x = (predictions[box_index + 0] + col) / side * w;
            boxes[index].y = (predictions[box_index + 1] + row) / side * h;
            boxes[index].w = pow(predictions[box_index + 2], (square?2:1)) * w;
            boxes[index].h = pow(predictions[box_index + 3], (square?2:1)) * h;
            for(j = 0; j < classes; ++j){
                int class_index = i*classes;
                float prob = scale*predictions[class_index+j];
                probs[index][j] = (prob > thresh) ? prob : 0;
            }
            if(only_objectness){
                probs[index][0] = scale;
            }
        }
    }
}

void print_yolo_detections(FILE **fps, char *id, box *boxes, float **probs, int total, int classes, int w, int h)
{
    int i, j;
    for(i = 0; i < total; ++i){
        float xmin = boxes[i].x - boxes[i].w/2.;
        float xmax = boxes[i].x + boxes[i].w/2.;
        float ymin = boxes[i].y - boxes[i].h/2.;
        float ymax = boxes[i].y + boxes[i].h/2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for(j = 0; j < classes; ++j){
            if (probs[i][j]) fprintf(fps[j], "%s %f %f %f %f %f\n", id, probs[i][j],
                    xmin, ymin, xmax, ymax);
        }
    }
}

void validate_yolo(char *cfgfile, char *weightfile)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    srand(time(0));

    char *base = "results/comp4_det_test_";
    //list *plist = get_paths("data/voc.2007.test");
    list *plist = get_paths("/home/pjreddie/data/voc/2007_test.txt");
    //list *plist = get_paths("data/voc.2012.test");
    char **paths = (char **)list_to_array(plist);

    layer l = net.layers[net.n-1];
    int classes = l.classes;
    int square = l.sqrt;
    int side = l.side;

    int j;
    FILE **fps = calloc(classes, sizeof(FILE *));
    for(j = 0; j < classes; ++j){
        char buff[1024];
        snprintf(buff, 1024, "%s%s.txt", base, voc_names[j]);
        fps[j] = fopen(buff, "w");
    }
    box *boxes = calloc(side*side*l.n, sizeof(box));
    float **probs = calloc(side*side*l.n, sizeof(float *));
    for(j = 0; j < side*side*l.n; ++j) probs[j] = calloc(classes, sizeof(float *));

    int m = plist->size;
    int i=0;
    int t;

    float thresh = .001;
    int nms = 1;
    float iou_thresh = .5;

    int nthreads = 2;
    image *val = calloc(nthreads, sizeof(image));
    image *val_resized = calloc(nthreads, sizeof(image));
    image *buf = calloc(nthreads, sizeof(image));
    image *buf_resized = calloc(nthreads, sizeof(image));
    pthread_t *thr = calloc(nthreads, sizeof(pthread_t));

    load_args args = {0};
    args.w = net.w;
    args.h = net.h;
    args.type = IMAGE_DATA;

    for(t = 0; t < nthreads; ++t){
        args.path = paths[i+t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    time_t start = time(0);
    for(i = nthreads; i < m+nthreads; i += nthreads){
        fprintf(stderr, "%d\n", i);
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for(t = 0; t < nthreads && i+t < m; ++t){
            args.path = paths[i+t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }
        for(t = 0; t < nthreads && i+t-nthreads < m; ++t){
            char *path = paths[i+t-nthreads];
            char *id = basecfg(path);
            float *X = val_resized[t].data;
            float *predictions = network_predict(net, X);
            int w = val[t].w;
            int h = val[t].h;
            convert_detections(predictions, classes, l.n, square, side, w, h, thresh, probs, boxes, 0);
            if (nms) do_nms_sort(boxes, probs, side*side*l.n, classes, iou_thresh);
            print_yolo_detections(fps, id, boxes, probs, side*side*l.n, classes, w, h);
            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }
    fprintf(stderr, "Total Detection Time: %f Seconds\n", (double)(time(0) - start));
}

void validate_yolo_recall(char *cfgfile, char *weightfile)
{
    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    srand(time(0));

    char *base = "results/comp4_det_test_";
    list *plist = get_paths("data/voc.2007.test");
    char **paths = (char **)list_to_array(plist);

    layer l = net.layers[net.n-1];
    int classes = l.classes;
    int square = l.sqrt;
    int side = l.side;

    int j, k;
    FILE **fps = calloc(classes, sizeof(FILE *));
    for(j = 0; j < classes; ++j){
        char buff[1024];
        snprintf(buff, 1024, "%s%s.txt", base, voc_names[j]);
        fps[j] = fopen(buff, "w");
    }
    box *boxes = calloc(side*side*l.n, sizeof(box));
    float **probs = calloc(side*side*l.n, sizeof(float *));
    for(j = 0; j < side*side*l.n; ++j) probs[j] = calloc(classes, sizeof(float *));

    int m = plist->size;
    int i=0;

    float thresh = .001;
    float iou_thresh = .5;
    float nms = 0;

    int total = 0;
    int correct = 0;
    int proposals = 0;
    float avg_iou = 0;

    for(i = 0; i < m; ++i){
        char *path = paths[i];
        image orig = load_image_color(path, 0, 0);
        image sized = resize_image(orig, net.w, net.h);
        char *id = basecfg(path);
        float *predictions = network_predict(net, sized.data);
        convert_detections(predictions, classes, l.n, square, side, 1, 1, thresh, probs, boxes, 1);
        if (nms) do_nms(boxes, probs, side*side*l.n, 1, nms);

        char *labelpath = find_replace(path, "images", "labels");
        labelpath = find_replace(labelpath, "JPEGImages", "labels");
        labelpath = find_replace(labelpath, ".jpg", ".txt");
        labelpath = find_replace(labelpath, ".JPEG", ".txt");

        int num_labels = 0;
        box_label *truth = read_boxes(labelpath, &num_labels);
        for(k = 0; k < side*side*l.n; ++k){
            if(probs[k][0] > thresh){
                ++proposals;
            }
        }
        for (j = 0; j < num_labels; ++j) {
            ++total;
            box t = {truth[j].x, truth[j].y, truth[j].w, truth[j].h};
            float best_iou = 0;
            for(k = 0; k < side*side*l.n; ++k){
                float iou = box_iou(boxes[k], t);
                if(probs[k][0] > thresh && iou > best_iou){
                    best_iou = iou;
                }
            }
            avg_iou += best_iou;
            if(best_iou > iou_thresh){
                ++correct;
            }
        }

        fprintf(stderr, "%5d %5d %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\n", i, correct, total, (float)proposals/(i+1), avg_iou*100/total, 100.*correct/total);
        free(id);
        free_image(orig);
        free_image(sized);
    }
}

void test_yolo(char *cfgfile, char *weightfile, char *filename, float thresh)
{

    network net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    detection_layer l = net.layers[net.n-1];
    set_batch_network(&net, 1);
    srand(2222222);
    clock_t time;
    char buff[256];
    char *input = buff;
    int j;
    float nms=.5;
    box *boxes = calloc(l.side*l.side*l.n, sizeof(box));
    float **probs = calloc(l.side*l.side*l.n, sizeof(float *));
    for(j = 0; j < l.side*l.side*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));
    while(1){
        if(filename){
            strncpy(input, filename, 256);
        } else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image_color(input,0,0);
        image sized = resize_image(im, net.w, net.h);
        float *X = sized.data;
        time=clock();
        float *predictions = network_predict(net, X);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        convert_detections(predictions, l.classes, l.n, l.sqrt, l.side, 1, 1, thresh, probs, boxes, 0);
        if (nms) do_nms_sort(boxes, probs, l.side*l.side*l.n, l.classes, nms);
        //draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, voc_names, voc_labels, 20);
        draw_detections(im, l.side*l.side*l.n, thresh, boxes, probs, voc_names, voc_labels, 20);
        save_image(im, "predictions");
        show_image(im, "predictions");

        show_image(sized, "resized");
        free_image(im);
        free_image(sized);
#ifdef OPENCV
        cvWaitKey(0);
        cvDestroyAllWindows();
#endif
        if (filename) break;
    }
}


//image load_image_from_socket(int sockfd){
    //TODO: write load_image_from_socket!
//  return image();
//}


void zpad(char* buffer, int n){
    int len = strlen(buffer);
    int i;
    for(i = 0; i < len; ++ i){
        buffer[n-i-1] = buffer[len - i - 1];
    }
    for(i = 0; i < n - len; ++ i){
        buffer[i] = '0';
    }
    buffer[n] = '\0';
}
static float **probs;
static box *boxes;
static network net;
static char inBuf[518400];
static char oBuf[518400];
static image in;
static image in_s;
static image det;
static image det_s;
static float fps = 0;
static int newsockfd1;
static int newsockfd2;

static int stop = 0;
pthread_t fetch_thread;
pthread_t detect_thread;
void *fetch_in_thread(void *ptr)
{
	char flag[10];
    int n;
    // printf("yolo fetch expecting y!\n");
    n = read(newsockfd1,&flag, 1);
    // printf("yolo fetch read y !\n");
    if(flag[0] != 'y' || n == 0){
        printf("ERROR: I don't know you %d!\n", n);
        stop = 1;
        return -1;
    }
    // printf("yolo read %s!\n", flag);
    FILE *fp;
    fp=fopen("/dev/shm/I_LOVE_CS", "r");
    fread((void*)&inBuf, 1, 518400, fp);
    fclose(fp);
    //printf("yolo read file\n");
    in = load_image_from_memory(&inBuf, 518400, 3);
    //printf("yolo load image\n");
    if(!in.data){
        printf("ERROR: memo load failed. \n");
        return -1;
    }
    in_s = resize_image(in, net.w, net.h);
    // printf("yolo fetch done\n");
    return 0;
}
void *detect_in_thread(void *ptr)
{
	//printf("yolo detecting!\n");
	int n;
    float nms = .4;
    detection_layer l = net.layers[net.n-1];
    if(!det_s.data){
    	printf("yolo detect bug!\n");
    	return -1;
    }
    float *X = det_s.data;
    float *predictions = network_predict(net, X);
    //printf("yolo predicted!\n");
    convert_detections(predictions, l.classes, l.n, l.sqrt, l.side, 1, 1, 0.2, probs, boxes, 0);
    if (nms > 0) do_nms(boxes, probs, l.side*l.side*l.n, l.classes, nms);
    // printf("\033[2J");
    // printf("\033[1;1H");
    // printf("\nFPS:%.0f\n",fps);
    // printf("Objects:\n\n");
    //printf("yolo draw\n");
    draw_detections(det, l.side*l.side*l.n, 0.2, boxes, probs, voc_names, voc_labels, 20);
    //printf("yolo draw done\n");
    int i,j,k,c=3;
    int h=360,w=480;
    for(k = 0; k < c; ++k){
        for(j = 0; j < h; ++j){
            for(i = 0; i < w; ++i){
                int dst_index = i + w*j + w*h*k;
                int src_index = k + c*i + c*w*j;
                oBuf[src_index] = floor(det.data[dst_index] * 255);
            }
        }
    }
    FILE *fp;
    fp=fopen("/dev/shm/ME_TOO", "w+");
    fwrite((void*)&oBuf, 1, 518400, fp);
    fclose(fp);
    n = write(newsockfd2,"y",1);
    //printf("yolo det done\n");
    if(n <= 0){
    	stop = 1;
    	printf("yolo det stop!\n");
    }
    return 0;
}
void server_yolo(char* cfgfile, char* weightfile, float thresh){
    //SOCKET RELATED VARIABLES
    int sockfd1, sockfd2, fd, n;
    socklen_t clilen1, clilen2;
    struct sockaddr_in serv_addr1, serv_addr2, cli_addr1, cli_addr2;
    //DNN RELATED VARIABLES
    net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&net, weightfile);
    }
    detection_layer l = net.layers[net.n-1];
    set_batch_network(&net,1);
    srand(19941002);
    clock_t time;
    char buffer[100000];
    float blockSize = 100000;
    char img[518400];
    char flag[10];
    int i,j;
    float nms = 0.5;
    boxes = calloc(l.side*l.side*l.n, sizeof(box));
    probs = calloc(l.side*l.side*l.n, sizeof(float *));
    for(j = 0; j < l.side*l.side*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));

    //SOCKET
    sockfd1 = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd1 < 0) printf("ERROR OPENING SOCKET 1");
    sockfd2 = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd2 < 0) printf("ERROR OPENING SOCKET 2");

    //BINDING
    bzero((char *) &serv_addr1, sizeof(serv_addr1));
    bzero((char *) &serv_addr2, sizeof(serv_addr2));
    serv_addr1.sin_family = AF_INET;
    serv_addr1.sin_addr.s_addr = INADDR_ANY;
    serv_addr1.sin_port = htons(1912);
    serv_addr2.sin_family = AF_INET;
    serv_addr2.sin_addr.s_addr = INADDR_ANY;
    serv_addr2.sin_port = htons(1913);
    if (bind(sockfd1, (struct sockaddr *) &serv_addr1, sizeof(serv_addr1)) < 0){
        printf("ERROR ON BINDING 1");
        return -1;
    }
    if (bind(sockfd2, (struct sockaddr *) &serv_addr2, sizeof(serv_addr2)) < 0){
        printf("ERROR ON BINDING 2");
        return -1;
    }
    listen(sockfd1,5);
    listen(sockfd2,5);
    clilen1 = sizeof(cli_addr1);
    clilen2 = sizeof(cli_addr2);
    fprintf(stderr, "Socket server start\n");
    struct timeval tval_last, tval_now;
    pthread_t fetch_thread;
    pthread_t detect_thread;
    int fresh;
    while(1)
    {
    	fresh = 1;
    	newsockfd1 = accept(sockfd1,(struct sockaddr *) &cli_addr1, &clilen1);
    	if (newsockfd1 < 0){
            printf("ERROR ON ACCEPT 1\n");
            continue;
        }
    	newsockfd2 = accept(sockfd2,(struct sockaddr *) &cli_addr2, &clilen2);
    	if (newsockfd2 < 0){
            printf("ERROR ON ACCEPT 2\n");
            continue;
        }
        printf("1&2 Connected OK!\n");
    	while(1){
	        // gettimeofday(&tval_now, NULL);
            // if(fps < 30){
            //     n = read(newsockfd1,&flag, 1);
            //     n = write(newsockfd2,"y",1);
            //     fps += 1;
            //     continue;
            // }
	        if(fresh)
	        {
	            //printf("OK! NEW START!");
                fps = 30;
	            fetch_in_thread(0);
                n = write(newsockfd2,"n",1);
	            //printf("yolo wrote n %d\n", n);
	            det = in;
	            det_s = in_s;
	            
	            if(n <= 0){
	            	close(newsockfd1);
	            	close(newsockfd2);
	            	break;
	            }
	            fresh = 0;
	        }
	        else
	        {
	            // struct timeval tval_before, tval_after, tval_result;
	            // gettimeofday(&tval_before, NULL);
	            //printf("start!\n");
	            if(pthread_create(&fetch_thread, 0, fetch_in_thread, 0)) printf("ERROR: Thread creation failed!\n");
	            if(pthread_create(&detect_thread, 0, detect_in_thread, 0)) printf("ERROR: Thread creation failed!\n");
	            pthread_join(fetch_thread, 0);
	            pthread_join(detect_thread, 0);
	            //printf("%s\n");
	            if(n <= 0) break;
	            if(det.data) free_image(det);
	            if(det_s.data) free_image(det_s);
	            //printf("joined!\n");
            	det = in;
            	det_s = in_s;
	            
	            // gettimeofday(&tval_after, NULL);
	            // timersub(&tval_after, &tval_before, &tval_result);
	            // float curr = 1000000.f/((long int)tval_result.tv_usec);
	            // fps = .9*fps + .1*curr;
	        }
	        tval_last = tval_now;
	        if(stop){
	        	close(newsockfd2);
	        	close(newsockfd1);
	        	stop = 0;
	        	break;
	        }
    	}
    }
    
}

void run_yolo(int argc, char **argv)
{
    int i;
    for(i = 0; i < 20; ++i){
        char buff[256];
        sprintf(buff, "data/labels/%s.png", voc_names[i]);
        voc_labels[i] = load_image_color(buff, 0, 0);
    }

    float thresh = find_float_arg(argc, argv, "-thresh", .2);
    int cam_index = find_int_arg(argc, argv, "-c", 0);
    int frame_skip = find_int_arg(argc, argv, "-s", 0);
    if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }

    char *cfg = argv[3];
    char *weights = (argc > 4) ? argv[4] : 0;
    char *filename = (argc > 5) ? argv[5]: 0;
    if(0==strcmp(argv[2], "test")) test_yolo(cfg, weights, filename, thresh);
    else if(0==strcmp(argv[2], "server")) server_yolo(cfg, weights,thresh);
    else if(0==strcmp(argv[2], "train")) train_yolo(cfg, weights);
    else if(0==strcmp(argv[2], "valid")) validate_yolo(cfg, weights);
    else if(0==strcmp(argv[2], "recall")) validate_yolo_recall(cfg, weights);
    else if(0==strcmp(argv[2], "demo")) demo(cfg, weights, thresh, cam_index, filename, voc_names, voc_labels, 20, frame_skip);
}





