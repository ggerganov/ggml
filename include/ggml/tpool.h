#pragma once

#if defined(_WIN32)
#include <windows.h>
#include "pthread-win32.h"
#else
#include <pthread.h>
#endif

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

    while (1) {
        pthread_mutex_lock(&tm->work_mutex);
        while (tm->work_first == NULL && !tm->stop)
            pthread_cond_wait(&tm->work_cond, &tm->work_mutex);
        if (tm->stop)
            break;
        work = tpool_work_get(tm);
        tm->working_cnt++;
        pthread_mutex_unlock(&tm->work_mutex);

        if (work != NULL) {
            work->func(work->arg);
            tpool_work_destroy(work);
        }

        pthread_mutex_lock(&tm->work_mutex);
        tm->working_cnt--;
        bool predicate = tm->working_cnt == 0 && tm->work_first == NULL;
        pthread_mutex_unlock(&tm->work_mutex);
        if(predicate)
            pthread_cond_signal(&tm->working_cond);
    }

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

    pthread_mutex_unlock(&tm->work_mutex);
    pthread_cond_broadcast(&tm->work_cond);

    return true;
}
