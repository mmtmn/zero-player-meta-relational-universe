#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <sstream>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

static const char* glErrToStr(GLenum err) {
    switch (err) {
        case GL_NO_ERROR: return "GL_NO_ERROR";
        case GL_INVALID_ENUM: return "GL_INVALID_ENUM";
        case GL_INVALID_VALUE: return "GL_INVALID_VALUE";
        case GL_INVALID_OPERATION: return "GL_INVALID_OPERATION";
        case GL_INVALID_FRAMEBUFFER_OPERATION: return "GL_INVALID_FRAMEBUFFER_OPERATION";
        case GL_OUT_OF_MEMORY: return "GL_OUT_OF_MEMORY";
        default: return "GL_UNKNOWN_ERROR";
    }
}

#define GL_CHECK() do { \
    GLenum _e = glGetError(); \
    if (_e != GL_NO_ERROR) { \
        fprintf(stderr, "OpenGL Error at %s:%d: %s (%u)\n", __FILE__, __LINE__, glErrToStr(_e), (unsigned)_e); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

static int gWidth  = 1280;
static int gHeight = 720;

static const int N = 48;
static const int E = (N * (N - 1)) / 2;

static const float DT = 0.010f;

static const float K_MEMORY   = 1.35f;
static const float GAMMA_META = 2.20f;

static const float ETA_REP    = 0.025f;
static const float DELTA_REP  = 0.020f;

static const float K_CM       = 2.00f;
static const float K_SCALE    = 6.50f;
static const float TARGET_RMS = 1.15f;

static const float M_X = 1.0f;
static const float M_Y = 6.0f;
static const float M_Z = 18.0f;

static const float M_S = 1.0f;

static const float DAMP_X = 0.20f;
static const float DAMP_S = 0.65f;

static const float K_SYM = 14.0f;
static const float K_ROW = 1.25f;

static const float EPS = 1e-20f;

static const float EDGE_ALPHA_MAX  = 0.85f;
static const float EDGE_ALPHA_GAIN = 2.60f;

static const float ACTION_GAIN     = 2.80f;

static glm::vec3 cameraPos   = glm::vec3(0.0f, 0.0f, 6.0f);
static glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
static glm::vec3 cameraUp    = glm::vec3(0.0f, 1.0f, 0.0f);

static float yaw   = -90.0f;
static float pitch =   0.0f;
static float fov   =  50.0f;

static float lastX = 0.0f;
static float lastY = 0.0f;
static bool  firstMouse = true;

static float deltaTime = 0.0f;
static float lastFrame = 0.0f;

static bool keys[1024] = { false };
static bool paused = false;

static void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    (void)window;
    if (width <= 0 || height <= 0) return;
    gWidth = width;
    gHeight = height;
    glViewport(0, 0, gWidth, gHeight);
}

static void mouse_callback(GLFWwindow* window, double xposIn, double yposIn) {
    (void)window;
    float xpos = (float)xposIn;
    float ypos = (float)yposIn;

    if (firstMouse) {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
        return;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos;
    lastX = xpos;
    lastY = ypos;

    float sensitivity = 0.10f;
    xoffset *= sensitivity;
    yoffset *= sensitivity;

    yaw   += xoffset;
    pitch += yoffset;

    if (pitch > 89.0f)  pitch = 89.0f;
    if (pitch < -89.0f) pitch = -89.0f;

    glm::vec3 front;
    front.x = cosf(glm::radians(yaw)) * cosf(glm::radians(pitch));
    front.y = sinf(glm::radians(pitch));
    front.z = sinf(glm::radians(yaw)) * cosf(glm::radians(pitch));
    cameraFront = glm::normalize(front);
}

static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    (void)window;
    (void)xoffset;
    fov -= (float)yoffset;
    if (fov < 12.0f) fov = 12.0f;
    if (fov > 90.0f) fov = 90.0f;
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    (void)window;
    (void)scancode;
    (void)mods;

    if (key >= 0 && key < 1024) {
        if (action == GLFW_PRESS) keys[key] = true;
        if (action == GLFW_RELEASE) keys[key] = false;
    }

    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
        paused = !paused;
}

static void processInput(float dt) {
    float speed = 5.0f * dt;
    glm::vec3 right = glm::normalize(glm::cross(cameraFront, cameraUp));

    if (keys[GLFW_KEY_W]) cameraPos += speed * cameraFront;
    if (keys[GLFW_KEY_S]) cameraPos -= speed * cameraFront;
    if (keys[GLFW_KEY_A]) cameraPos -= speed * right;
    if (keys[GLFW_KEY_D]) cameraPos += speed * right;
    if (keys[GLFW_KEY_Q]) cameraPos -= speed * cameraUp;
    if (keys[GLFW_KEY_E]) cameraPos += speed * cameraUp;
}

struct Vertex {
    float px, py, pz;
    float cr, cg, cb, ca;
};

static cudaGraphicsResource_t cudaVboPoints = nullptr;
static cudaGraphicsResource_t cudaVboLines  = nullptr;

static GLuint compileShader(GLenum type, const char* src) {
    GLuint sh = glCreateShader(type);
    glShaderSource(sh, 1, &src, nullptr);
    glCompileShader(sh);

    GLint ok = 0;
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetShaderInfoLog(sh, sizeof(log), nullptr, log);
        fprintf(stderr, "Shader compile failed: %s\n", log);
        exit(EXIT_FAILURE);
    }
    return sh;
}

static GLuint linkProgram(GLuint vs, GLuint fs) {
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);

    GLint ok = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetProgramInfoLog(prog, sizeof(log), nullptr, log);
        fprintf(stderr, "Program link failed: %s\n", log);
        exit(EXIT_FAILURE);
    }
    return prog;
}

__device__ __forceinline__ float clampf(float x, float a, float b) {
    return fminf(fmaxf(x, a), b);
}

__device__ __forceinline__ void atomicAdd3(float* fx, float* fy, float* fz, float ax, float ay, float az) {
    atomicAdd(fx, ax);
    atomicAdd(fy, ay);
    atomicAdd(fz, az);
}

__device__ __forceinline__ void hsv2rgb(float h, float s, float v, float& r, float& g, float& b) {
    h = h - floorf(h);
    float c = v * s;
    float x = c * (1.0f - fabsf(fmodf(h * 6.0f, 2.0f) - 1.0f));
    float m = v - c;

    float rp=0.0f, gp=0.0f, bp=0.0f;
    float hp = h * 6.0f;

    if (hp < 1.0f)      { rp=c; gp=x; bp=0.0f; }
    else if (hp < 2.0f) { rp=x; gp=c; bp=0.0f; }
    else if (hp < 3.0f) { rp=0.0f; gp=c; bp=x; }
    else if (hp < 4.0f) { rp=0.0f; gp=x; bp=c; }
    else if (hp < 5.0f) { rp=x; gp=0.0f; bp=c; }
    else                { rp=c; gp=0.0f; bp=x; }

    r = rp + m;
    g = gp + m;
    b = bp + m;
}

__global__ void zero_array(float* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    a[i] = 0.0f;
}

__global__ void compute_row_stats_S(const float* S, float* rowMaxS, float* rowLogSumExpS) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float mx = -1.0e30f;
    int base = i * N;
    for (int j = 0; j < N; ++j) {
        if (j == i) continue;
        float v = S[base + j];
        mx = fmaxf(mx, v);
    }

    float sum = 0.0f;
    for (int j = 0; j < N; ++j) {
        if (j == i) continue;
        float v = S[base + j];
        sum += expf(v - mx);
    }
    sum = fmaxf(sum, EPS);

    rowMaxS[i] = mx;
    rowLogSumExpS[i] = logf(sum);
}

__global__ void compute_row_mean_S(const float* S, float* rowMeanS) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int base = i * N;
    float s = 0.0f;
    for (int j = 0; j < N; ++j) {
        if (j == i) continue;
        s += S[base + j];
    }
    rowMeanS[i] = s / (float)(N - 1);
}

__global__ void compute_row_stats_T(const float* S, const float* X, const float* rowMaxS, const float* rowLogSumExpS, float* rowMaxT, float* rowLogSumExpT) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float xi = X[i*3+0], yi = X[i*3+1], zi = X[i*3+2];
    float maxS = rowMaxS[i];
    float logSumS = rowLogSumExpS[i];

    float mx = -1.0e30f;
    int base = i * N;

    for (int j = 0; j < N; ++j) {
        if (j == i) continue;
        float xj = X[j*3+0], yj = X[j*3+1], zj = X[j*3+2];
        float dx = xi - xj;
        float dy = yi - yj;
        float dz = zi - zj;
        float dist2 = dx*dx + dy*dy + dz*dz;

        float logP = (S[base + j] - maxS) - logSumS;
        float Tij = K_MEMORY * logP - GAMMA_META * dist2;
        mx = fmaxf(mx, Tij);
    }

    float sum = 0.0f;
    for (int j = 0; j < N; ++j) {
        if (j == i) continue;
        float xj = X[j*3+0], yj = X[j*3+1], zj = X[j*3+2];
        float dx = xi - xj;
        float dy = yi - yj;
        float dz = zi - zj;
        float dist2 = dx*dx + dy*dy + dz*dz;

        float logP = (S[base + j] - maxS) - logSumS;
        float Tij = K_MEMORY * logP - GAMMA_META * dist2;
        sum += expf(Tij - mx);
    }
    sum = fmaxf(sum, EPS);

    rowMaxT[i] = mx;
    rowLogSumExpT[i] = logf(sum);
}

__global__ void compute_row_kl_entropy(const float* S, const float* X, const float* rowMaxS, const float* rowLogSumExpS, const float* rowMaxT, const float* rowLogSumExpT, float* rowKL, float* rowEntropy) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float xi = X[i*3+0], yi = X[i*3+1], zi = X[i*3+2];
    float maxS = rowMaxS[i];
    float logSumS = rowLogSumExpS[i];
    float maxT = rowMaxT[i];
    float logSumT = rowLogSumExpT[i];

    int base = i * N;
    float KL = 0.0f;
    float H = 0.0f;

    for (int j = 0; j < N; ++j) {
        if (j == i) continue;

        float xj = X[j*3+0], yj = X[j*3+1], zj = X[j*3+2];
        float dx = xi - xj;
        float dy = yi - yj;
        float dz = zi - zj;
        float dist2 = dx*dx + dy*dy + dz*dz;

        float logP = (S[base + j] - maxS) - logSumS;
        float P = expf(logP);

        float T = K_MEMORY * logP - GAMMA_META * dist2;
        float logQ = (T - maxT) - logSumT;

        KL += P * (logP - logQ);
        H  += -P * logP;
    }

    rowKL[i] = fmaxf(KL, 0.0f);
    rowEntropy[i] = fmaxf(H, 0.0f);
}

__global__ void reduce_cm_r2(const float* X, float* outCM, float* outR2) {
    __shared__ float sCMx[256];
    __shared__ float sCMy[256];
    __shared__ float sCMz[256];
    __shared__ float sR2[256];

    int tid = threadIdx.x;
    float cmx = 0.0f, cmy = 0.0f, cmz = 0.0f, r2 = 0.0f;

    for (int i = tid; i < N; i += blockDim.x) {
        float x = X[i*3+0];
        float y = X[i*3+1];
        float z = X[i*3+2];
        cmx += x;
        cmy += y;
        cmz += z;
        r2  += x*x + y*y + z*z;
    }

    sCMx[tid] = cmx;
    sCMy[tid] = cmy;
    sCMz[tid] = cmz;
    sR2[tid]  = r2;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sCMx[tid] += sCMx[tid + stride];
            sCMy[tid] += sCMy[tid + stride];
            sCMz[tid] += sCMz[tid + stride];
            sR2[tid]  += sR2[tid  + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        outCM[0] = sCMx[0];
        outCM[1] = sCMy[0];
        outCM[2] = sCMz[0];
        outR2[0] = sR2[0];
    }
}

__global__ void apply_gauge_forces_X(const float* X, float* FX, const float* CM, const float* sumR2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float cmx = CM[0] / (float)N;
    float cmy = CM[1] / (float)N;
    float cmz = CM[2] / (float)N;

    float meanR2 = sumR2[0] / (float)N;
    float scaleErr = meanR2 - (TARGET_RMS * TARGET_RMS);
    float scaleCoef = -K_SCALE * scaleErr * (2.0f / (float)N);

    float x = X[i*3+0];
    float y = X[i*3+1];
    float z = X[i*3+2];

    FX[i*3+0] += -K_CM * cmx + scaleCoef * x;
    FX[i*3+1] += -K_CM * cmy + scaleCoef * y;
    FX[i*3+2] += -K_CM * cmz + scaleCoef * z;
}

__global__ void compute_edge_forces(
    const float* S,
    const float* X,
    const float* rowMaxS,
    const float* rowLogSumExpS,
    const float* rowMaxT,
    const float* rowLogSumExpT,
    const float* rowMeanS,
    const float* rowKL,
    float* FX,
    float* FS,
    const int* edgeU,
    const int* edgeV
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;

    int i = edgeU[e];
    int j = edgeV[e];

    int ij = i * N + j;
    int ji = j * N + i;

    float xi = X[i*3+0], yi = X[i*3+1], zi = X[i*3+2];
    float xj = X[j*3+0], yj = X[j*3+1], zj = X[j*3+2];

    float dx = xi - xj;
    float dy = yi - yj;
    float dz = zi - zj;
    float dist2 = dx*dx + dy*dy + dz*dz;

    float logPij = (S[ij] - rowMaxS[i]) - rowLogSumExpS[i];
    float logPji = (S[ji] - rowMaxS[j]) - rowLogSumExpS[j];

    float Pij = expf(logPij);
    float Pji = expf(logPji);

    float Tij = K_MEMORY * logPij - GAMMA_META * dist2;
    float Tji = K_MEMORY * logPji - GAMMA_META * dist2;

    float logQij = (Tij - rowMaxT[i]) - rowLogSumExpT[i];
    float logQji = (Tji - rowMaxT[j]) - rowLogSumExpT[j];

    float Qij = expf(logQij);
    float Qji = expf(logQji);

    float Aij = logPij - logQij;
    float Aji = logPji - logQji;

    float Vi = rowKL[i];
    float Vj = rowKL[j];

    float dVdSij = Pij * (Aij - Vi) + K_MEMORY * (Qij - Pij);
    float dVdSji = Pji * (Aji - Vj) + K_MEMORY * (Qji - Pji);

    float sym = S[ij] - S[ji];

    float rowFix_i = -K_ROW * (rowMeanS[i] / (float)(N - 1));
    float rowFix_j = -K_ROW * (rowMeanS[j] / (float)(N - 1));

    float Fij = -dVdSij - K_SYM * sym + rowFix_i;
    float Fji = -dVdSji + K_SYM * sym + rowFix_j;

    FS[ij] = Fij;
    FS[ji] = Fji;

    float coefMeta = 2.0f * GAMMA_META * ((Qij - Pij) + (Qji - Pji));
    float fx = coefMeta * dx;
    float fy = coefMeta * dy;
    float fz = coefMeta * dz;

    float denom = dist2 + DELTA_REP;
    float repScale = ETA_REP / (denom * denom);
    fx += repScale * dx;
    fy += repScale * dy;
    fz += repScale * dz;

    atomicAdd3(&FX[i*3+0], &FX[i*3+1], &FX[i*3+2], fx, fy, fz);
    atomicAdd3(&FX[j*3+0], &FX[j*3+1], &FX[j*3+2], -fx, -fy, -fz);
}

__global__ void integrate_half_X(float* X, float* V, const float* FX) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float vx = V[i*3+0];
    float vy = V[i*3+1];
    float vz = V[i*3+2];

    vx += (FX[i*3+0] / M_X) * (0.5f * DT);
    vy += (FX[i*3+1] / M_Y) * (0.5f * DT);
    vz += (FX[i*3+2] / M_Z) * (0.5f * DT);

    float x = X[i*3+0] + vx * DT;
    float y = X[i*3+1] + vy * DT;
    float z = X[i*3+2] + vz * DT;

    V[i*3+0] = vx;
    V[i*3+1] = vy;
    V[i*3+2] = vz;

    X[i*3+0] = x;
    X[i*3+1] = y;
    X[i*3+2] = z;
}

__global__ void integrate_finish_X(float* V, const float* FX) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float vx = V[i*3+0];
    float vy = V[i*3+1];
    float vz = V[i*3+2];

    vx += (FX[i*3+0] / M_X) * (0.5f * DT);
    vy += (FX[i*3+1] / M_Y) * (0.5f * DT);
    vz += (FX[i*3+2] / M_Z) * (0.5f * DT);

    V[i*3+0] = vx;
    V[i*3+1] = vy;
    V[i*3+2] = vz;
}

__global__ void integrate_half_S(float* S, float* Sd, const float* FS) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * N;
    if (idx >= total) return;

    int i = idx / N;
    int j = idx - i * N;
    if (i == j) {
        S[idx] = 0.0f;
        Sd[idx] = 0.0f;
        return;
    }

    float sd = Sd[idx];
    sd += (FS[idx] / M_S) * (0.5f * DT);
    float s = S[idx] + sd * DT;

    Sd[idx] = sd;
    S[idx] = s;
}

__global__ void integrate_finish_S(float* Sd, const float* FS) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * N;
    if (idx >= total) return;

    int i = idx / N;
    int j = idx - i * N;
    if (i == j) {
        Sd[idx] = 0.0f;
        return;
    }

    float sd = Sd[idx];
    sd += (FS[idx] / M_S) * (0.5f * DT);
    Sd[idx] = sd;
}

__global__ void apply_damping_X(float* V, float damp) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    V[i*3+0] *= damp;
    V[i*3+1] *= damp;
    V[i*3+2] *= damp;
}

__global__ void apply_damping_S(float* Sd, float damp) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * N;
    if (idx >= total) return;

    int i = idx / N;
    int j = idx - i * N;
    if (i == j) {
        Sd[idx] = 0.0f;
        return;
    }

    Sd[idx] *= damp;
}

__global__ void fill_points_vbo(Vertex* outPts, const float* X, const float* action, const float* entropy, float t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float x = X[i*3+0];
    float y = X[i*3+1];
    float z = X[i*3+2];

    float a = action[i];
    float heat = 1.0f - expf(-ACTION_GAIN * fmaxf(a, 0.0f));

    float H = entropy[i];
    float Hmax = logf((float)(N - 1));
    float hnorm = clampf(H / fmaxf(Hmax, 1e-6f), 0.0f, 1.0f);

    float hue = 0.68f - 0.62f * (1.0f - hnorm);
    hue += 0.02f * sinf(0.7f * t + 0.31f * (float)i);

    float sat = 0.30f + 0.70f * heat;
    float val = 0.65f + 0.35f * heat;

    float r, g, b;
    hsv2rgb(hue, sat, val, r, g, b);

    Vertex v;
    v.px = x;
    v.py = y;
    v.pz = z;
    v.cr = r;
    v.cg = g;
    v.cb = b;
    v.ca = 1.0f;

    outPts[i] = v;
}

__global__ void fill_lines_vbo(Vertex* outLines, const float* X, const float* S, const float* rowMaxS, const float* rowLogSumExpS, const int* edgeU, const int* edgeV, float t) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;

    int i = edgeU[e];
    int j = edgeV[e];

    float xi = X[i*3+0], yi = X[i*3+1], zi = X[i*3+2];
    float xj = X[j*3+0], yj = X[j*3+1], zj = X[j*3+2];

    float logPij = (S[i*N + j] - rowMaxS[i]) - rowLogSumExpS[i];
    float logPji = (S[j*N + i] - rowMaxS[j]) - rowLogSumExpS[j];

    float pij = expf(logPij);
    float pji = expf(logPji);

    float w = 0.5f * (pij + pji);
    float tt = clampf(w * 3.5f, 0.0f, 1.0f);

    float hue = 0.64f - 0.58f * tt;
    hue += 0.02f * sinf(0.35f * t + 0.013f * (float)e);

    float sat = 0.65f + 0.25f * tt;
    float val = 0.55f + 0.45f * tt;

    float r, g, b;
    hsv2rgb(hue, sat, val, r, g, b);

    float alpha = powf(fmaxf(w, 0.0f), 0.60f) * EDGE_ALPHA_GAIN;
    alpha = clampf(alpha, 0.0f, EDGE_ALPHA_MAX);

    Vertex v0, v1;
    v0.px = xi; v0.py = yi; v0.pz = zi;
    v1.px = xj; v1.py = yj; v1.pz = zj;

    v0.cr = r; v0.cg = g; v0.cb = b; v0.ca = alpha;
    v1.cr = r; v1.cg = g; v1.cb = b; v1.ca = alpha;

    outLines[2*e + 0] = v0;
    outLines[2*e + 1] = v1;
}

static void createVboVao(GLuint& vao, GLuint& vbo, size_t bytes) {
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    glBufferData(GL_ARRAY_BUFFER, bytes, nullptr, GL_DYNAMIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(3 * sizeof(float)));

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    GL_CHECK();
}

static float effectiveDimensionFromPositions(const std::vector<float>& hX) {
    glm::vec3 mean(0.0f);
    for (int i = 0; i < N; ++i) {
        mean.x += hX[i*3+0];
        mean.y += hX[i*3+1];
        mean.z += hX[i*3+2];
    }
    mean /= (float)N;

    float C[3][3] = {};
    for (int i = 0; i < N; ++i) {
        float x = hX[i*3+0] - mean.x;
        float y = hX[i*3+1] - mean.y;
        float z = hX[i*3+2] - mean.z;
        C[0][0] += x*x; C[0][1] += x*y; C[0][2] += x*z;
        C[1][0] += y*x; C[1][1] += y*y; C[1][2] += y*z;
        C[2][0] += z*x; C[2][1] += z*y; C[2][2] += z*z;
    }
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            C[r][c] /= (float)N;

    float A[3][3] = {
        {C[0][0], C[0][1], C[0][2]},
        {C[1][0], C[1][1], C[1][2]},
        {C[2][0], C[2][1], C[2][2]},
    };

    auto jacobiRotate = [&](int p, int q) {
        if (fabsf(A[p][q]) < 1e-12f) return;
        float app = A[p][p];
        float aqq = A[q][q];
        float apq = A[p][q];

        float phi = 0.5f * atanf(2.0f * apq / (aqq - app + 1e-12f));
        float c = cosf(phi);
        float s = sinf(phi);

        float app2 = c*c*app - 2.0f*s*c*apq + s*s*aqq;
        float aqq2 = s*s*app + 2.0f*s*c*apq + c*c*aqq;

        A[p][p] = app2;
        A[q][q] = aqq2;
        A[p][q] = 0.0f;
        A[q][p] = 0.0f;

        for (int k = 0; k < 3; ++k) {
            if (k == p || k == q) continue;
            float akp = A[k][p];
            float akq = A[k][q];
            A[k][p] = c*akp - s*akq;
            A[p][k] = A[k][p];
            A[k][q] = s*akp + c*akq;
            A[q][k] = A[k][q];
        }
    };

    for (int sweep = 0; sweep < 16; ++sweep) {
        jacobiRotate(0,1);
        jacobiRotate(0,2);
        jacobiRotate(1,2);
    }

    float l1 = fmaxf(A[0][0], 0.0f);
    float l2 = fmaxf(A[1][1], 0.0f);
    float l3 = fmaxf(A[2][2], 0.0f);

    float s1 = l1 + l2 + l3;
    float s2 = l1*l1 + l2*l2 + l3*l3;
    if (s2 < 1e-16f) return 0.0f;
    return (s1*s1) / s2;
}

static void reset_sim(
    float* dS, float* dSd,
    float* dX, float* dV,
    const std::vector<int>& hU,
    const std::vector<int>& hV,
    int* dEdgeU,
    int* dEdgeV
) {
    std::vector<float> hX(N*3, 0.0f);
    std::vector<float> hVel(N*3, 0.0f);

    for (int i = 0; i < N; ++i) {
        float t = (float)i / (float)(N - 1);
        float x = (t * 2.0f - 1.0f) * 1.2f;
        float y = 1e-6f * sinf(12.9898f * (float)i);
        float z = 1e-6f * cosf(78.233f  * (float)i);
        hX[i*3+0] = x;
        hX[i*3+1] = y;
        hX[i*3+2] = z;
    }

    std::vector<float> hS(N*N, 0.0f);
    std::vector<float> hSd(N*N, 0.0f);

    float S0 = -2.60f;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i == j) hS[i*N + j] = 0.0f;
            else        hS[i*N + j] = S0;
        }
    }

    CUDA_CHECK(cudaMemcpy(dX,  hX.data(),  N*3*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV,  hVel.data(), N*3*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dS,  hS.data(),  N*N*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dSd, hSd.data(), N*N*sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(dEdgeU, hU.data(), E*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dEdgeV, hV.data(), E*sizeof(int), cudaMemcpyHostToDevice));
}

int main() {
    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);

    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    if (!monitor) {
        fprintf(stderr, "No primary monitor found\n");
        glfwTerminate();
        return -1;
    }
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);
    if (!mode) {
        fprintf(stderr, "No video mode found\n");
        glfwTerminate();
        return -1;
    }

    gWidth = mode->width;
    gHeight = mode->height;

    GLFWwindow* window = glfwCreateWindow(gWidth, gHeight, "Meta Relational Universe", monitor, nullptr);
    if (!window) {
        fprintf(stderr, "Failed to create GLFW window\n");
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetKeyCallback(window, key_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    lastX = gWidth * 0.5f;
    lastY = gHeight * 0.5f;

    glewExperimental = GL_TRUE;
    GLenum glewErr = glewInit();
    if (glewErr != GLEW_OK) {
        fprintf(stderr, "GLEW init failed: %s\n", glewGetErrorString(glewErr));
        return -1;
    }

    glViewport(0, 0, gWidth, gHeight);

    unsigned int cudaDeviceCount = 0;
    int cudaDevices[16] = {};
    cudaError_t interopErr = cudaGLGetDevices(&cudaDeviceCount, cudaDevices, 16, cudaGLDeviceListAll);
    if (interopErr == cudaSuccess && cudaDeviceCount > 0) {
        CUDA_CHECK(cudaSetDevice(cudaDevices[0]));
    } else {
        CUDA_CHECK(cudaSetDevice(0));
    }
    CUDA_CHECK(cudaFree(0));

    const char* bgVS = R"GLSL(
        #version 330 core
        out vec2 vUV;
        void main() {
            vec2 pos;
            if (gl_VertexID == 0) pos = vec2(-1.0, -1.0);
            if (gl_VertexID == 1) pos = vec2( 3.0, -1.0);
            if (gl_VertexID == 2) pos = vec2(-1.0,  3.0);
            vUV = 0.5 * (pos + 1.0);
            gl_Position = vec4(pos, 0.0, 1.0);
        }
    )GLSL";

    const char* bgFS = R"GLSL(
        #version 330 core
        in vec2 vUV;
        out vec4 FragColor;

        uniform vec2 uResolution;
        uniform float uTime;

        void main() {
            vec2 uv = gl_FragCoord.xy / uResolution;
            vec3 top = vec3(0.06, 0.08, 0.16);
            vec3 bot = vec3(0.01, 0.01, 0.04);
            float g = smoothstep(0.0, 1.0, uv.y);
            float wig = 0.008 * sin(uTime * 0.12 + uv.x * 6.0);
            vec3 col = mix(bot, top, clamp(g + wig, 0.0, 1.0));
            vec2 q = uv * 2.0 - 1.0;
            col *= 1.0 - 0.22 * dot(q, q);
            FragColor = vec4(col, 1.0);
        }
    )GLSL";

    GLuint bgV = compileShader(GL_VERTEX_SHADER, bgVS);
    GLuint bgF = compileShader(GL_FRAGMENT_SHADER, bgFS);
    GLuint bgProg = linkProgram(bgV, bgF);
    glDeleteShader(bgV);
    glDeleteShader(bgF);

    GLint bgRes = glGetUniformLocation(bgProg, "uResolution");
    GLint bgTime = glGetUniformLocation(bgProg, "uTime");

    const char* vtxSrc = R"GLSL(
        #version 330 core
        layout(location = 0) in vec3 aPos;
        layout(location = 1) in vec4 aColor;

        uniform mat4 uMVP;
        uniform float uPointSize;

        out vec4 vColor;

        void main() {
            gl_Position = uMVP * vec4(aPos, 1.0);
            gl_PointSize = uPointSize;
            vColor = aColor;
        }
    )GLSL";

    const char* lineFS = R"GLSL(
        #version 330 core
        in vec4 vColor;
        out vec4 FragColor;
        void main() {
            FragColor = vColor;
        }
    )GLSL";

    const char* pointFS = R"GLSL(
        #version 330 core
        in vec4 vColor;
        out vec4 FragColor;
        void main() {
            vec2 p = gl_PointCoord * 2.0 - 1.0;
            float r2 = dot(p, p);
            if (r2 > 1.0) discard;

            float glow = exp(-3.0 * r2);
            vec3 col = clamp(vColor.rgb * (0.28 + 0.92 * glow), 0.0, 1.0);
            float a = clamp(vColor.a * (0.20 + 0.80 * glow), 0.0, 1.0);

            FragColor = vec4(col, a);
        }
    )GLSL";

    GLuint vtx = compileShader(GL_VERTEX_SHADER, vtxSrc);
    GLuint lfs = compileShader(GL_FRAGMENT_SHADER, lineFS);
    GLuint pfs = compileShader(GL_FRAGMENT_SHADER, pointFS);

    GLuint lineProg = linkProgram(vtx, lfs);
    GLuint pointProg = linkProgram(vtx, pfs);

    glDeleteShader(vtx);
    glDeleteShader(lfs);
    glDeleteShader(pfs);

    GLint uMVP_line  = glGetUniformLocation(lineProg, "uMVP");
    GLint uMVP_point = glGetUniformLocation(pointProg, "uMVP");
    GLint uPointSize = glGetUniformLocation(pointProg, "uPointSize");

    glEnable(GL_MULTISAMPLE);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glEnable(GL_PROGRAM_POINT_SIZE);

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    GLuint vaoPts, vboPts;
    GLuint vaoLines, vboLines;
    createVboVao(vaoPts, vboPts, sizeof(Vertex) * (size_t)N);
    createVboVao(vaoLines, vboLines, sizeof(Vertex) * (size_t)(2 * E));

    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cudaVboPoints, vboPts, cudaGraphicsRegisterFlagsWriteDiscard));
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cudaVboLines,  vboLines, cudaGraphicsRegisterFlagsWriteDiscard));

    std::vector<int> hU(E), hV(E);
    int eidx = 0;
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            hU[eidx] = i;
            hV[eidx] = j;
            eidx++;
        }
    }

    float *dS, *dSd, *dX, *dV;
    float *dRowMaxS, *dRowLogSumExpS, *dRowMeanS;
    float *dRowMaxT, *dRowLogSumExpT;
    float *dAction, *dEntropy;
    float *dFX, *dFS;
    float *dCM, *dSumR2;
    int *dEdgeU, *dEdgeV;

    CUDA_CHECK(cudaMalloc(&dS,               N*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dSd,              N*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dX,               N*3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dV,               N*3*sizeof(float)));

    CUDA_CHECK(cudaMalloc(&dRowMaxS,         N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dRowLogSumExpS,   N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dRowMeanS,        N*sizeof(float)));

    CUDA_CHECK(cudaMalloc(&dRowMaxT,         N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dRowLogSumExpT,   N*sizeof(float)));

    CUDA_CHECK(cudaMalloc(&dAction,          N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dEntropy,         N*sizeof(float)));

    CUDA_CHECK(cudaMalloc(&dFX,              N*3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dFS,              N*N*sizeof(float)));

    CUDA_CHECK(cudaMalloc(&dCM,              3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dSumR2,           1*sizeof(float)));

    CUDA_CHECK(cudaMalloc(&dEdgeU,           E*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dEdgeV,           E*sizeof(int)));

    reset_sim(dS, dSd, dX, dV, hU, hV, dEdgeU, dEdgeV);

    std::vector<float> hX(N*3, 0.0f);
    double titleAccum = 0.0;

    GLuint bgVAO = 0;
    glGenVertexArrays(1, &bgVAO);

    auto computeForcesAndStats = [&]() {
        compute_row_stats_S<<<1, 128>>>(dS, dRowMaxS, dRowLogSumExpS);
        compute_row_mean_S<<<1, 128>>>(dS, dRowMeanS);
        compute_row_stats_T<<<1, 128>>>(dS, dX, dRowMaxS, dRowLogSumExpS, dRowMaxT, dRowLogSumExpT);
        compute_row_kl_entropy<<<1, 128>>>(dS, dX, dRowMaxS, dRowLogSumExpS, dRowMaxT, dRowLogSumExpT, dAction, dEntropy);

        zero_array<<<1, 256>>>(dFX, N*3);
        zero_array<<<(N*N + 255)/256, 256>>>(dFS, N*N);

        compute_edge_forces<<<(E + 255)/256, 256>>>(
            dS, dX,
            dRowMaxS, dRowLogSumExpS,
            dRowMaxT, dRowLogSumExpT,
            dRowMeanS,
            dAction,
            dFX, dFS,
            dEdgeU, dEdgeV
        );

        reduce_cm_r2<<<1, 256>>>(dX, dCM, dSumR2);
        apply_gauge_forces_X<<<1, 128>>>(dX, dFX, dCM, dSumR2);

        CUDA_CHECK(cudaGetLastError());
    };

    float dampX = expf(-DAMP_X * DT);
    float dampS = expf(-DAMP_S * DT);

    while (!glfwWindowShouldClose(window)) {
        float currentFrame = (float)glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        glfwPollEvents();
        processInput(deltaTime);

        if (keys[GLFW_KEY_R]) {
            reset_sim(dS, dSd, dX, dV, hU, hV, dEdgeU, dEdgeV);
            keys[GLFW_KEY_R] = false;
        }

        if (!paused) {
            int substeps = (int)fminf(8.0f, fmaxf(1.0f, deltaTime / DT));
            for (int s = 0; s < substeps; ++s) {
                computeForcesAndStats();

                integrate_half_X<<<1, 128>>>(dX, dV, dFX);
                integrate_half_S<<<(N*N + 255)/256, 256>>>(dS, dSd, dFS);

                computeForcesAndStats();

                integrate_finish_X<<<1, 128>>>(dV, dFX);
                integrate_finish_S<<<(N*N + 255)/256, 256>>>(dSd, dFS);

                apply_damping_X<<<1, 128>>>(dV, dampX);
                apply_damping_S<<<(N*N + 255)/256, 256>>>(dSd, dampS);

                CUDA_CHECK(cudaGetLastError());
            }
        } else {
            computeForcesAndStats();
        }

        titleAccum += (double)deltaTime;
        if (titleAccum > 0.25) {
            titleAccum = 0.0;
            CUDA_CHECK(cudaMemcpy(hX.data(), dX, N*3*sizeof(float), cudaMemcpyDeviceToHost));
            float deff = effectiveDimensionFromPositions(hX);

            std::ostringstream oss;
            oss.setf(std::ios::fixed);
            oss.precision(2);
            oss << "Meta Relational Universe | d_eff " << deff << (paused ? " | PAUSED" : "");
            glfwSetWindowTitle(window, oss.str().c_str());
        }

        CUDA_CHECK(cudaGraphicsMapResources(1, &cudaVboPoints, 0));
        CUDA_CHECK(cudaGraphicsMapResources(1, &cudaVboLines,  0));

        Vertex* dPts = nullptr;
        Vertex* dLines = nullptr;
        size_t ptsBytes = 0, linesBytes = 0;

        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&dPts, &ptsBytes, cudaVboPoints));
        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&dLines, &linesBytes, cudaVboLines));

        float t = (float)glfwGetTime();

        fill_points_vbo<<<1, 128>>>(dPts, dX, dAction, dEntropy, t);
        fill_lines_vbo<<<(E + 255)/256, 256>>>(dLines, dX, dS, dRowMaxS, dRowLogSumExpS, dEdgeU, dEdgeV, t);

        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaVboPoints, 0));
        CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaVboLines,  0));

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glDisable(GL_DEPTH_TEST);
        glDepthMask(GL_FALSE);
        glDisable(GL_BLEND);

        glUseProgram(bgProg);
        glUniform2f(bgRes, (float)gWidth, (float)gHeight);
        glUniform1f(bgTime, (float)glfwGetTime());
        glBindVertexArray(bgVAO);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glBindVertexArray(0);

        glm::mat4 proj = glm::perspective(glm::radians(fov), (float)gWidth/(float)gHeight, 0.05f, 250.0f);
        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
        glm::mat4 mvp  = proj * view;

        glEnable(GL_BLEND);
        glEnable(GL_DEPTH_TEST);
        glDepthMask(GL_FALSE);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE);

        glUseProgram(lineProg);
        glUniformMatrix4fv(uMVP_line, 1, GL_FALSE, glm::value_ptr(mvp));
        glBindVertexArray(vaoLines);
        glDrawArrays(GL_LINES, 0, 2 * E);
        glBindVertexArray(0);

        glDisable(GL_DEPTH_TEST);
        glDepthMask(GL_FALSE);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        glUseProgram(pointProg);
        glUniformMatrix4fv(uMVP_point, 1, GL_FALSE, glm::value_ptr(mvp));
        glUniform1f(uPointSize, 11.0f);
        glBindVertexArray(vaoPts);
        glDrawArrays(GL_POINTS, 0, N);
        glBindVertexArray(0);

        glDepthMask(GL_TRUE);

        glfwSwapBuffers(window);
    }

    CUDA_CHECK(cudaGraphicsUnregisterResource(cudaVboPoints));
    CUDA_CHECK(cudaGraphicsUnregisterResource(cudaVboLines));

    glDeleteVertexArrays(1, &bgVAO);

    glDeleteBuffers(1, &vboPts);
    glDeleteVertexArrays(1, &vaoPts);
    glDeleteBuffers(1, &vboLines);
    glDeleteVertexArrays(1, &vaoLines);

    glDeleteProgram(bgProg);
    glDeleteProgram(lineProg);
    glDeleteProgram(pointProg);

    CUDA_CHECK(cudaFree(dS));
    CUDA_CHECK(cudaFree(dSd));
    CUDA_CHECK(cudaFree(dX));
    CUDA_CHECK(cudaFree(dV));

    CUDA_CHECK(cudaFree(dRowMaxS));
    CUDA_CHECK(cudaFree(dRowLogSumExpS));
    CUDA_CHECK(cudaFree(dRowMeanS));

    CUDA_CHECK(cudaFree(dRowMaxT));
    CUDA_CHECK(cudaFree(dRowLogSumExpT));

    CUDA_CHECK(cudaFree(dAction));
    CUDA_CHECK(cudaFree(dEntropy));

    CUDA_CHECK(cudaFree(dFX));
    CUDA_CHECK(cudaFree(dFS));

    CUDA_CHECK(cudaFree(dCM));
    CUDA_CHECK(cudaFree(dSumR2));

    CUDA_CHECK(cudaFree(dEdgeU));
    CUDA_CHECK(cudaFree(dEdgeV));

    glfwTerminate();
    return 0;
}
