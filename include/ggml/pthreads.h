#include <stdbool.h>
#include <time.h>
#include <windows.h>

static DWORD timespec_to_ms(const struct timespec* abstime)
{
    DWORD t;

    if (abstime == NULL)
        return INFINITE;

    t = ((abstime->tv_sec - time(NULL)) * 1000) + (abstime->tv_nsec / 1000000);
    if (t < 0)
        t = 1;
    return t;
}

static void ms_to_timespec(struct timespec* ts, unsigned int ms)
{
    if (ts == NULL)
        return;
    ts->tv_sec = (ms / 1000) + time(NULL);
    ts->tv_nsec = (ms % 1000) * 1000000;
}

typedef HANDLE pthread_t;
typedef void pthread_attr_t;
typedef DWORD thread_ret_t;

typedef struct {
    void *(*start_routine)(void *);
    void *start_arg;
} win_thread_start_t;

static DWORD WINAPI win_thread_start(void *arg)
{
    win_thread_start_t *data       = arg;
    void *(*start_routine)(void *) = data->start_routine;
    void *start_arg                = data->start_arg;

    free(data);

    start_routine(start_arg);
    return 0; /* ERROR_SUCCESS */
}

static int pthread_create(pthread_t *thread, pthread_attr_t *attr, void *(*start_routine)(void *), void *arg)
{
    win_thread_start_t *data;

    if (thread == NULL || start_routine == NULL)
        return 1;

    data = malloc(sizeof(*data));
    data->start_routine = start_routine;
    data->start_arg     = arg;

    *thread = CreateThread(NULL, 0, win_thread_start, data, 0, NULL);
    if (*thread == NULL)
        return 1;
    return 0;
}

static int pthread_join(pthread_t thread, void **value_ptr)
{
    (void)value_ptr;
    WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
    return 0;
}

static int pthread_detach(pthread_t thread)
{
    CloseHandle(thread);
}

typedef CRITICAL_SECTION pthread_mutex_t;
typedef void pthread_mutexattr_t;

static int pthread_mutex_init(pthread_mutex_t *mutex, pthread_mutexattr_t *attr)
{
    (void)attr;

    if (mutex == NULL)
        return 1;

    InitializeCriticalSection(mutex);
    return 0;
}

static int pthread_mutex_destroy(pthread_mutex_t *mutex)
{
    if (mutex == NULL)
        return 1;
    DeleteCriticalSection(mutex);
    return 0;
}

static int pthread_mutex_lock(pthread_mutex_t *mutex)
{
    if (mutex == NULL)
        return 1;
    EnterCriticalSection(mutex);
    return 0;
}

static int pthread_mutex_unlock(pthread_mutex_t *mutex)
{
    if (mutex == NULL)
        return 1;
    LeaveCriticalSection(mutex);
    return 0;
}

typedef CONDITION_VARIABLE pthread_cond_t;
typedef void pthread_condattr_t;

#ifdef NEEDED
struct timespec {
    long tv_sec;
    long tv_nsec;
};
#endif

static int pthread_cond_init(pthread_cond_t *cond, pthread_condattr_t *attr)
{
    (void)attr;
    if (cond == NULL)
        return 1;
    InitializeConditionVariable(cond);
    return 0;
}

static int pthread_cond_destroy(pthread_cond_t *cond)
{
    /* Windows does not have a destroy for conditionals */
    (void)cond;
    return 0;
}

static int pthread_cond_timedwait(pthread_cond_t *cond, pthread_mutex_t *mutex,
        const struct timespec *abstime)
{
    if (cond == NULL || mutex == NULL)
        return 1;
    if (!SleepConditionVariableCS(cond, mutex, timespec_to_ms(abstime)))
        return 1;
    return 0;
}

static int pthread_cond_wait(pthread_cond_t *cond, pthread_mutex_t *mutex)
{
    if (cond == NULL || mutex == NULL)
        return 1;
    return pthread_cond_timedwait(cond, mutex, NULL);
}

static int pthread_cond_signal(pthread_cond_t *cond)
{
    if (cond == NULL)
        return 1;
    WakeConditionVariable(cond);
    return 0;
}

static int pthread_cond_broadcast(pthread_cond_t *cond)
{
    if (cond == NULL)
        return 1;
    WakeAllConditionVariable(cond);
    return 0;
}

typedef struct {
    SRWLOCK lock;
    bool    exclusive;
} pthread_rwlock_t;

typedef void pthread_rwlockattr_t;

static int pthread_rwlock_init(pthread_rwlock_t *rwlock, const pthread_rwlockattr_t *attr)
{
    (void)attr;
    if (rwlock == NULL)
        return 1;
    InitializeSRWLock(&rwlock->lock);
    rwlock->exclusive = false;
    return 0;
}

static int pthread_rwlock_destroy(pthread_rwlock_t *rwlock)
{
    (void)rwlock;
}

static int pthread_rwlock_rdlock(pthread_rwlock_t *rwlock)
{
    if (rwlock == NULL)
        return 1;
    AcquireSRWLockShared(&rwlock->lock);
}

static int pthread_rwlock_tryrdlock(pthread_rwlock_t *rwlock)
{
    if (rwlock == NULL)
        return 1;
    return !TryAcquireSRWLockShared(&rwlock->lock);
}

static int pthread_rwlock_wrlock(pthread_rwlock_t *rwlock)
{
    if (rwlock == NULL)
        return 1;
    AcquireSRWLockExclusive(&rwlock->lock);
    rwlock->exclusive = true;
}

static int pthread_rwlock_trywrlock(pthread_rwlock_t  *rwlock)
{
    BOOLEAN ret;

    if (rwlock == NULL)
        return 1;

    ret = TryAcquireSRWLockExclusive(&rwlock->lock);
    if (ret)
        rwlock->exclusive = true;
    return ret;
}

static int pthread_rwlock_unlock(pthread_rwlock_t *rwlock)
{
    if (rwlock == NULL)
        return 1;

    if (rwlock->exclusive) {
        rwlock->exclusive = false;
        ReleaseSRWLockExclusive(&rwlock->lock);
    } else {
        ReleaseSRWLockShared(&rwlock->lock);
    }
}

static unsigned int pcthread_get_num_procs()
{
    SYSTEM_INFO sysinfo;

    GetSystemInfo(&sysinfo);
    return sysinfo.dwNumberOfProcessors;
}

typedef void (*thread_func_t)(void *arg);

struct tpool_work {
    thread_func_t      func;
    void              *arg;
    struct tpool_work *next;
};
typedef struct tpool_work tpool_work_t;

static tpool_work_t *tpool_work_create(thread_func_t func, void *arg)
{
    tpool_work_t *work;

    if (func == NULL)
        return NULL;

    work       = malloc(sizeof(*work));
    work->func = func;
    work->arg  = arg;
    work->next = NULL;
    return work;
}

static void tpool_work_destroy(tpool_work_t *work)
{
    if (work == NULL)
        return;
    free(work);
}

struct tpool {
    tpool_work_t    *work_first;
    tpool_work_t    *work_last;
    pthread_mutex_t  work_mutex;
    pthread_cond_t   work_cond;
    pthread_cond_t   working_cond;
    size_t           working_cnt;
    size_t           thread_cnt;
    bool             stop;
};
typedef struct tpool tpool_t;

static tpool_work_t *tpool_work_get(tpool_t *tm)
{
    tpool_work_t *work;

    if (tm == NULL)
        return NULL;

    work = tm->work_first;
    if (work == NULL)
        return NULL;

    if (work->next == NULL) {
        tm->work_first = NULL;
        tm->work_last  = NULL;
    } else {
        tm->work_first = work->next;
    }

    return work;
}

static void *tpool_worker(void *arg)
{
    tpool_t      *tm = arg;
    tpool_work_t *work;

    // printf("pthreads %p starts\n", arg);
    while (1) {
        pthread_mutex_lock(&tm->work_mutex);
        while (tm->work_first == NULL && !tm->stop)
            pthread_cond_wait(&tm->work_cond, &tm->work_mutex);
        if (tm->stop)
            break;
        work = tpool_work_get(tm);
        tm->working_cnt++;
        pthread_mutex_unlock(&tm->work_mutex);

        // printf("pthreads %p works\n", arg);
        if (work != NULL) {
            work->func(work->arg);
            tpool_work_destroy(work);
        }
        // printf("pthreads %p waits\n", arg);

        pthread_mutex_lock(&tm->work_mutex);
        tm->working_cnt--;
        if (tm->working_cnt == 0 && tm->work_first == NULL)
            pthread_cond_signal(&tm->working_cond);
        pthread_mutex_unlock(&tm->work_mutex);
    }

    // printf("pthreads %p stops\n", arg);
    tm->thread_cnt--;
    pthread_mutex_unlock(&tm->work_mutex);
    pthread_cond_signal(&tm->working_cond);
    return NULL;
}

static tpool_t *tpool_create(size_t num)
{
    tpool_t   *tm;
    pthread_t  thread;
    size_t     i;

    if (num == 0)
        num = 2;

    tm             = calloc(1, sizeof(*tm));
    tm->thread_cnt = num;

    pthread_mutex_init(&tm->work_mutex, NULL);
    pthread_cond_init(&tm->work_cond, NULL);
    pthread_cond_init(&tm->working_cond, NULL);

    tm->work_first = NULL;
    tm->work_last  = NULL;

    for (i=0; i<num; i++) {
        pthread_create(&thread, NULL, tpool_worker, tm);
        pthread_detach(thread);
    }

    return tm;
}

static void tpool_wait(tpool_t *tm)
{
    if (tm == NULL)
        return;

    pthread_mutex_lock(&tm->work_mutex);
    while (tm->working_cnt != 0 || tm->work_first != NULL) {
        pthread_cond_wait(&tm->working_cond, &tm->work_mutex);
    }
    pthread_mutex_unlock(&tm->work_mutex);
}

static void tpool_destroy(tpool_t *tm)
{
    tpool_work_t *work;
    tpool_work_t *work2;

    if (tm == NULL)
        return;

    pthread_mutex_lock(&tm->work_mutex);
    work = tm->work_first;
    while (work != NULL) {
        work2 = work->next;
        tpool_work_destroy(work);
        work = work2;
    }
    tm->stop = true;
    pthread_mutex_unlock(&tm->work_mutex);
    pthread_cond_broadcast(&tm->work_cond);

    tpool_wait(tm);

    pthread_mutex_lock(&tm->work_mutex);
    while (tm->thread_cnt > 0)
        pthread_cond_wait(&tm->working_cond, &tm->work_mutex);
    pthread_mutex_unlock(&tm->work_mutex);

    pthread_mutex_destroy(&tm->work_mutex);
    pthread_cond_destroy(&tm->work_cond);
    pthread_cond_destroy(&tm->working_cond);

    free(tm);
}

static bool tpool_add_work(tpool_t *tm, thread_func_t func, void *arg)
{
    tpool_work_t *work;

    if (tm == NULL)
        return false;

    work = tpool_work_create(func, arg);
    if (work == NULL)
        return false;

    pthread_mutex_lock(&tm->work_mutex);
    if (tm->work_first == NULL) {
        tm->work_first = work;
        tm->work_last  = tm->work_first;
    } else {
        tm->work_last->next = work;
        tm->work_last       = work;
    }

    pthread_cond_broadcast(&tm->work_cond);
    pthread_mutex_unlock(&tm->work_mutex);

    return true;
}
