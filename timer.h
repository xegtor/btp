#include <sys/time.h>
#include "cpucounters.h"
#include "utils.h"
#include <iostream>

using namespace std;

PCM *___pcm;
SystemCounterState ___before_sstate, ___after_sstate;
double ___joules;
    
static void pcm_init() {
    ___pcm = PCM::getInstance();
    if (false) {
        ___pcm->resetPMU();
    }
    // program() creates common semaphore for the singleton, so ideally to be called before any other references to PCM
    PCM::ErrorCode status = ___pcm->program();

    switch (status)
    {
    case PCM::Success:
	cerr << "Success initializing PCM" << endl;
        break;
    case PCM::MSRAccessDenied:
        cerr << "Access to Processor Counter Monitor has denied (no MSR or PCI CFG space access)." << endl;
        exit(EXIT_FAILURE);
    case PCM::PMUBusy:
        cerr << "Access to Processor Counter Monitor has denied (Performance Monitoring Unit is occupied by other application). Try to stop the application that uses PMU." << endl;
        cerr << "Alternatively you can try running PCM with option -r to reset PMU configuration at your own risk." << endl;
        exit(EXIT_FAILURE);
    default:
        cerr << "Access to Processor Counter Monitor has denied (Unknown error)." << endl;
        exit(EXIT_FAILURE);
    }
    print_cpu_details();
}

#define PCM_INIT        { pcm_init(); }
#define PCM_START       { ___before_sstate = getSystemCounterState(); }
#define PCM_JOULES      { ___joules = getConsumedJoules(___before_sstate, ___after_sstate); }
#define PCM_STOP        { ___after_sstate = getSystemCounterState();  PCM_JOULES }
#define PCM_FINALIZE    { ___pcm->cleanup(); }

static long get_usecs (void)
{  
   struct timeval t;
   gettimeofday(&t,NULL);
   return t.tv_sec*1000000+t.tv_usec;
}

static long ___start, ___end, ___startCPU, ___endCPU, ___startGPU, ___endGPU;
static double ___dur=0, ___durCPU=0, ___durGPU=0;

#define REPORT_TIME	{double _dur_ = ((double)(___end-___start))/1000000; ___dur+=_dur_;}
#define REPORT_TIME_CPU	{double _dur_ = ((double)(___endCPU-___startCPU))/1000000; ___durCPU+=_dur_;}
#define REPORT_TIME_GPU	{double _dur_ = ((double)(___endGPU-___startGPU))/1000000; ___durGPU+=_dur_;}
#define PRINT_HARNESS_FOOTER { 		\
	printf("Harness Ended...\n");	\
	printf("============================ MMTk Statistics Totals ============================\n"); \
	printf("time.mu\ttime.cpu\ttime.gpu\tenergy\n");		\
	printf("%.3f\t%.3f\t%.3f\t%f\n",___dur,___durCPU,___durGPU,___joules);		\
	printf("Total time: %.3f ms\n", ___dur*1000);	\
	printf("------------------------------ End MMTk Statistics -----------------------------\n");	\
	printf("===== TEST PASSED in 0 msec =====\n");	\
}

#define START_TIMER	{___start = get_usecs();}
#define END_TIMER	{___end = get_usecs();REPORT_TIME;}
#define START_TIMER_CPU	{___startCPU = get_usecs();}
#define END_TIMER_CPU	{___endCPU = get_usecs();REPORT_TIME_CPU;}
#define START_TIMER_GPU	{___startGPU = get_usecs();}
#define END_TIMER_GPU	{___endGPU = get_usecs();REPORT_TIME_GPU;}
