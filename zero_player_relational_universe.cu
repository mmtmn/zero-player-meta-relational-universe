// zero_player_relational_universe.cu
// ============================================================
// High quality CUDA + OpenGL interop "zero player" relational universe
//
// Core idea (implemented, not just talked about):
// - Relations are the ontology: S_ij are unconstrained relation log strengths
//   R_ij = exp(S_ij) are positive relation strengths
// - Geometry is a field X_i in R^3, not assumed "given" but evolved as a coupled field
// - Dynamics follow from a single potential energy V (least action with Rayleigh dissipation)
//   d/dt(∂T/∂qdot) + ∂D/∂qdot + ∂V/∂q = 0
//
// No randomness, no schedulers, no “unlock dimension” flags.
// What looks like pruning emerges from the distance cost and material cost.
//
// Rendering:
// - Modern OpenGL 3.3 core
// - CUDA writes directly into OpenGL VBOs via cudaGraphicsGLRegisterBuffer
// - Camera controls: mouse look, WASD move, scroll FOV (based on your example style)
//
// Build:
//   nvcc -O3 zero_player_relational_universe.cu -lglfw -lGL -lGLEW -o universe
//
// Run:
//   ./universe
//
// Controls:
// - Mouse: look
// - WASD: move
// - Q/E: move down/up
// - Space: pause physics
// - R: reset physics
// - Esc: quit
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <sstream>

// ---- GLEW MUST COME FIRST ----
#include <GL/glew.h>

// ---- Then GLFW (brings in OpenGL) ----
#include <GLFW/glfw3.h>

// ---- CUDA AFTER OpenGL headers ----
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// ---- GLM is header-only, safe anywhere ----
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>


// ============================================================
// Error checking macros
// ============================================================

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

// ============================================================
// Simulation + render config
// ============================================================

static const int WIDTH  = 1280;
static const int HEIGHT = 800;

static const int N = 48;                         // entities
static const int E = (N * (N - 1)) / 2;           // undirected edges i<j

// Physics time step
static const float DT = 0.010f;

// Potential weights (tuneable, but deterministic)
static const float LAMBDA_DEG = 6.0f;            // degree budget constraint weight
static const float MU_MAT     = 0.25f;           // material cost on relations
static const float GAMMA_GEO  = 2.0f;            // geometry coupling (short edges preferred)
static const float ETA_REP    = 0.010f;          // repulsion (prevents collapse)
static const float DELTA_REP  = 0.010f;          // repulsion softening
static const float K_CM       = 2.0f;            // center of mass gauge fixing
static const float K_SCALE    = 6.0f;            // scale gauge fixing
static const float TARGET_RMS = 1.2f;            // target RMS radius

// Rayleigh dissipation (physical damping, still zero player)
static const float DAMP_S = 0.35f;               // relation velocity damping
static const float DAMP_X = 0.20f;               // geometry velocity damping

// Anisotropic masses create natural time-scale separation (1D looks stable first)
static const float M_X = 1.0f;
static const float M_Y = 6.0f;
static const float M_Z = 20.0f;

// Numeric safety for exp(S)
static const float S_MIN = -12.0f;               // exp(-12) ~ 6e-6
static const float S_MAX =  4.0f;                // exp(4) ~ 54

// Render mapping for edge alpha
static const float EDGE_ALPHA_GAIN = 0.18f;      // alpha = 1 - exp(-gain*R)
static const float EDGE_ALPHA_MAX  = 0.65f;

// ============================================================
// Camera + input state (based on your style)
// ============================================================

static glm::vec3 cameraPos   = glm::vec3(0.0f, 0.0f, 6.0f);
static glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
static glm::vec3 cameraUp    = glm::vec3(0.0f, 1.0f, 0.0f);

static float yaw   = -90.0f;
static float pitch =   0.0f;
static float fov   =  45.0f;

static float lastX = WIDTH * 0.5f;
static float lastY = HEIGHT * 0.5f;
static bool  firstMouse = true;

static float deltaTime = 0.0f;
static float lastFrame = 0.0f;

static bool keys[1024] = { false };
static bool paused = false;

// ============================================================
// CUDA <-> OpenGL interop resources
// ============================================================

struct Vertex {
    float px, py, pz;      // position
    float cr, cg, cb, ca;  // color RGBA
};

static cudaGraphicsResource_t cudaVboPoints = nullptr;
static cudaGraphicsResource_t cudaVboLines  = nullptr;

// ============================================================
// OpenGL helpers: shader compilation
// ============================================================

static GLuint compileShader(GLenum type, const char* src) {
    GLuint sh = glCreateShader(type);
    glShaderSource(sh, 1, &src, nullptr);
    glCompileShader(sh);

    GLint ok = 0;
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048];
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
        char log[2048];
        glGetProgramInfoLog(prog, sizeof(log), nullptr, log);
        fprintf(stderr, "Program link failed: %s\n", log);
        exit(EXIT_FAILURE);
    }
    return prog;
}

// ============================================================
// Callbacks (mouse, scroll, keyboard) with camera controls
// ============================================================

static void mouse_callback(GLFWwindow* window, double xposIn, double yposIn) {
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

    if (pitch > 89.0f) pitch = 89.0f;
    if (pitch < -89.0f) pitch = -89.0f;

    glm::vec3 front;
    front.x = cosf(glm::radians(yaw)) * cosf(glm::radians(pitch));
    front.y = sinf(glm::radians(pitch));
    front.z = sinf(glm::radians(yaw)) * cosf(glm::radians(pitch));
    cameraFront = glm::normalize(front);
}

static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    (void)xoffset;
    fov -= (float)yoffset;
    if (fov < 10.0f) fov = 10.0f;
    if (fov > 90.0f) fov = 90.0f;
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
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

// ============================================================
// CUDA device helpers
// ============================================================

__device__ __forceinline__ float clampf(float x, float a, float b) {
    return fminf(fmaxf(x, a), b);
}

__device__ __forceinline__ float safeExp(float x) {
    x = clampf(x, S_MIN, S_MAX);
    return expf(x);
}

__device__ __forceinline__ void atomicAdd3(float* fx, float* fy, float* fz, float ax, float ay, float az) {
    atomicAdd(fx, ax);
    atomicAdd(fy, ay);
    atomicAdd(fz, az);
}

// ============================================================
// CUDA kernels: simulation
// State:
// - S: NxN symmetric, diag unused. R = exp(S)
// - Sd: NxN symmetric velocities for S
// - X: Nx3 positions
// - V: Nx3 velocities
// - deg: N vector of degrees deg_i = sum_j R_ij
// - b: N vector target degree budget
// - edgeU, edgeV: E edges listing i<j
// ============================================================

__global__ void compute_degree(const float* S, float* deg) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float sum = 0.0f;
    int base = i * N;
    for (int j = 0; j < N; ++j) {
        if (j == i) continue;
        float Rij = safeExp(S[base + j]);
        sum += Rij;
    }
    deg[i] = sum;
}

__global__ void zero_forces(float* FX) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N * 3) return;
    FX[i] = 0.0f;
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

__global__ void edge_forces_and_update_S(
    float* S, float* Sd,
    const float* X,
    const float* deg,
    const float* b,
    const int* edgeU,
    const int* edgeV,
    float* FX
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;

    int i = edgeU[e];
    int j = edgeV[e];

    int ij = i * N + j;
    int ji = j * N + i;

    float Si = S[ij];
    float Rij = safeExp(Si);

    float xi = X[i*3+0], yi = X[i*3+1], zi = X[i*3+2];
    float xj = X[j*3+0], yj = X[j*3+1], zj = X[j*3+2];

    float dx = xi - xj;
    float dy = yi - yj;
    float dz = zi - zj;

    float dist2 = dx*dx + dy*dy + dz*dz + 1e-12f;

    // Potential components:
    // V_deg = (lambda/2) sum_i (deg_i - b_i)^2
    // V_mat = (mu/2) sum_ij R_ij^2
    // V_geo = (gamma/2) sum_ij R_ij * ||Xi - Xj||^2
    //
    // dV/dR_ij = lambda[(deg_i - b_i) + (deg_j - b_j)] + mu*R_ij + (gamma/2)*dist2
    float dVdR = LAMBDA_DEG * ((deg[i] - b[i]) + (deg[j] - b[j]))
               + MU_MAT * Rij
               + 0.5f * GAMMA_GEO * dist2;

    // Since R = exp(S), dR/dS = R, so dV/dS = dV/dR * R
    float dVdS = dVdR * Rij;

    // Equation with Rayleigh dissipation: Sddot + DAMP_S * Sdot + dV/dS = 0
    // Using symplectic-ish explicit update:
    float accS = -dVdS - DAMP_S * Sd[ij];

    float Sd_new = Sd[ij] + accS * DT;
    float S_new  = S[ij]  + Sd_new * DT;

    // clamp to avoid exp blowup
    S_new = clampf(S_new, S_MIN, S_MAX);

    Sd[ij] = Sd_new;
    Sd[ji] = Sd_new;
    S[ij]  = S_new;
    S[ji]  = S_new;

    // Geometry forces:
    // From V_geo: F = -∂V/∂X_i = -gamma * R_ij * (Xi - Xj)
    float fx = -GAMMA_GEO * Rij * dx;
    float fy = -GAMMA_GEO * Rij * dy;
    float fz = -GAMMA_GEO * Rij * dz;

    // Repulsion:
    // V_rep = (eta/2) sum 1/(dist2 + delta)
    // Force magnitude: +eta * (Xi - Xj) / (dist2 + delta)^2
    float denom = (dist2 + DELTA_REP);
    float repScale = ETA_REP / (denom * denom);
    fx += repScale * dx;
    fy += repScale * dy;
    fz += repScale * dz;

    // Accumulate forces atomically
    atomicAdd3(&FX[i*3+0], &FX[i*3+1], &FX[i*3+2], fx, fy, fz);
    atomicAdd3(&FX[j*3+0], &FX[j*3+1], &FX[j*3+2], -fx, -fy, -fz);
}

__global__ void integrate_X(
    float* X, float* V,
    const float* FX,
    const float* CM,
    const float* sumR2
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float cmx = CM[0] / (float)N;
    float cmy = CM[1] / (float)N;
    float cmz = CM[2] / (float)N;

    float meanR2 = sumR2[0] / (float)N;
    float scaleErr = meanR2 - (TARGET_RMS * TARGET_RMS);

    float x = X[i*3+0];
    float y = X[i*3+1];
    float z = X[i*3+2];

    float fx = FX[i*3+0];
    float fy = FX[i*3+1];
    float fz = FX[i*3+2];

    // Gauge fixing forces
    // Center of mass
    fx += -K_CM * cmx;
    fy += -K_CM * cmy;
    fz += -K_CM * cmz;

    // Scale
    float scaleCoef = -K_SCALE * scaleErr * (2.0f / (float)N);
    fx += scaleCoef * x;
    fy += scaleCoef * y;
    fz += scaleCoef * z;

    // Anisotropic masses
    float ax = fx / M_X;
    float ay = fy / M_Y;
    float az = fz / M_Z;

    // Rayleigh dissipation: Xddot + DAMP_X * Xdot + dV/dX = 0
    float vx = V[i*3+0];
    float vy = V[i*3+1];
    float vz = V[i*3+2];

    ax += -DAMP_X * vx;
    ay += -DAMP_X * vy;
    az += -DAMP_X * vz;

    vx += ax * DT;
    vy += ay * DT;
    vz += az * DT;

    x  += vx * DT;
    y  += vy * DT;
    z  += vz * DT;

    V[i*3+0] = vx;
    V[i*3+1] = vy;
    V[i*3+2] = vz;

    X[i*3+0] = x;
    X[i*3+1] = y;
    X[i*3+2] = z;
}

// ============================================================
// CUDA kernels: write to OpenGL VBOs (interop)
// ============================================================

__global__ void fill_points_vbo(Vertex* outPts, const float* X, const float* deg, const float* b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float x = X[i*3+0];
    float y = X[i*3+1];
    float z = X[i*3+2];

    float err = fabsf(deg[i] - b[i]);
    float heat = 1.0f - expf(-1.25f * err);

    Vertex v;
    v.px = x;
    v.py = y;
    v.pz = z;

    // White to warm color based on constraint stress
    v.cr = 0.90f + 0.10f * heat;
    v.cg = 0.90f - 0.35f * heat;
    v.cb = 0.95f - 0.55f * heat;
    v.ca = 1.00f;

    outPts[i] = v;
}

__global__ void fill_lines_vbo(Vertex* outLines, const float* X, const float* S, const int* edgeU, const int* edgeV) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;

    int i = edgeU[e];
    int j = edgeV[e];

    float xi = X[i*3+0], yi = X[i*3+1], zi = X[i*3+2];
    float xj = X[j*3+0], yj = X[j*3+1], zj = X[j*3+2];

    float Rij = safeExp(S[i*N + j]);
    float a = 1.0f - expf(-EDGE_ALPHA_GAIN * Rij);
    a = clampf(a, 0.0f, EDGE_ALPHA_MAX);

    // Cyan-ish edges
    float cr = 0.25f;
    float cg = 0.85f;
    float cb = 1.00f;

    Vertex v0, v1;
    v0.px = xi; v0.py = yi; v0.pz = zi;
    v1.px = xj; v1.py = yj; v1.pz = zj;

    v0.cr = cr; v0.cg = cg; v0.cb = cb; v0.ca = a;
    v1.cr = cr; v1.cg = cg; v1.cb = cb; v1.ca = a;

    outLines[2*e + 0] = v0;
    outLines[2*e + 1] = v1;
}

// ============================================================
// Host helpers: create VBO + VAO for Vertex
// ============================================================

static void createVboVao(GLuint& vao, GLuint& vbo, size_t bytes) {
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    glBufferData(GL_ARRAY_BUFFER, bytes, nullptr, GL_DYNAMIC_DRAW);

    // layout(location=0) vec3 pos
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);

    // layout(location=1) vec4 color
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(3 * sizeof(float)));

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    GL_CHECK();
}

// ============================================================
// Process input for camera movement
// ============================================================

static void processInput(float dt) {
    float speed = 4.0f * dt;
    glm::vec3 right = glm::normalize(glm::cross(cameraFront, cameraUp));

    if (keys[GLFW_KEY_W]) cameraPos += speed * cameraFront;
    if (keys[GLFW_KEY_S]) cameraPos -= speed * cameraFront;
    if (keys[GLFW_KEY_A]) cameraPos -= speed * right;
    if (keys[GLFW_KEY_D]) cameraPos += speed * right;
    if (keys[GLFW_KEY_Q]) cameraPos -= speed * cameraUp;
    if (keys[GLFW_KEY_E]) cameraPos += speed * cameraUp;
}

// ============================================================
// Effective dimension estimate (CPU) for window title
// Uses covariance eigenvalues participation ratio in 3D
// d_eff = (sum λ)^2 / (sum λ^2), ranges 1..3
// ============================================================

static float effectiveDimensionFromPositions(const std::vector<float>& hX) {
    glm::vec3 mean(0.0f);
    for (int i = 0; i < N; ++i) {
        mean.x += hX[i*3+0];
        mean.y += hX[i*3+1];
        mean.z += hX[i*3+2];
    }
    mean /= (float)N;

    // covariance
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

    // Eigenvalues of 3x3 symmetric matrix (closed form is messy; do power iterations for 3 modes)
    // For title display only, do simple Jacobi sweeps
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

// ============================================================
// Reset simulation to deterministic initial conditions
// ============================================================

static void reset_sim(
    float* dS, float* dSd,
    float* dX, float* dV,
    float* dB,
    const std::vector<int>& hU,
    const std::vector<int>& hV,
    int* dU,
    int* dVedge
) {
    // Target degree budget b_i (positive, consistent with R>0)
    std::vector<float> hB(N, 1.0f);
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), N*sizeof(float), cudaMemcpyHostToDevice));

    // Deterministic initial geometry: nearly 1D line with tiny deterministic transverse perturbation
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
    CUDA_CHECK(cudaMemcpy(dX, hX.data(), N*3*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, hVel.data(), N*3*sizeof(float), cudaMemcpyHostToDevice));

    // Deterministic initial relations: uniform log-strength (complete graph)
    // Start moderate so constraints + geometry decide structure
    std::vector<float> hS(N*N, S_MIN);
    std::vector<float> hSd(N*N, 0.0f);

    float R0 = 0.05f;               // exp(log(R0))
    float S0 = logf(R0);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i == j) {
                hS[i*N + j] = S_MIN;
            } else {
                hS[i*N + j] = S0;
            }
        }
    }

    CUDA_CHECK(cudaMemcpy(dS,  hS.data(),  N*N*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dSd, hSd.data(), N*N*sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(dU, hU.data(), E*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dVedge, hV.data(), E*sizeof(int), cudaMemcpyHostToDevice));
}

// ============================================================
// Main
// ============================================================

int main() {
    // GLFW init
    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "Zero Player Relational Universe", nullptr, nullptr);
    if (!window) {
        fprintf(stderr, "Failed to create GLFW window\n");
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetKeyCallback(window, key_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // GLEW init
    glewExperimental = GL_TRUE;
    GLenum glewErr = glewInit();
    if (glewErr != GLEW_OK) {
        fprintf(stderr, "GLEW init failed: %s\n", glewGetErrorString(glewErr));
        return -1;
    }

    // CUDA GL device
    CUDA_CHECK(cudaGLSetGLDevice(0));

    // Shaders
    const char* vsSrc = R"GLSL(
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

    const char* fsSrc = R"GLSL(
        #version 330 core
        in vec4 vColor;
        out vec4 FragColor;
        void main() {
            FragColor = vColor;
        }
    )GLSL";

    GLuint vs = compileShader(GL_VERTEX_SHADER, vsSrc);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, fsSrc);
    GLuint prog = linkProgram(vs, fs);
    glDeleteShader(vs);
    glDeleteShader(fs);

    GLint uMVP = glGetUniformLocation(prog, "uMVP");
    GLint uPointSize = glGetUniformLocation(prog, "uPointSize");

    // GL state
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glClearColor(0.03f, 0.03f, 0.06f, 1.0f);

    // Create VBO/VAO for points and lines
    GLuint vaoPts, vboPts;
    GLuint vaoLines, vboLines;

    createVboVao(vaoPts, vboPts, sizeof(Vertex) * (size_t)N);
    createVboVao(vaoLines, vboLines, sizeof(Vertex) * (size_t)(2 * E));

    // Register VBOs with CUDA
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cudaVboPoints, vboPts, cudaGraphicsRegisterFlagsWriteDiscard));
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cudaVboLines,  vboLines, cudaGraphicsRegisterFlagsWriteDiscard));

    // Build fixed edge list i<j
    std::vector<int> hU(E), hV(E);
    int idx = 0;
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            hU[idx] = i;
            hV[idx] = j;
            idx++;
        }
    }

    // Allocate simulation buffers on GPU
    float *dS, *dSd, *dX, *dV, *dDeg, *dB, *dFX;
    float *dCM, *dSumR2;
    int *dEdgeU, *dEdgeV;

    CUDA_CHECK(cudaMalloc(&dS,    N*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dSd,   N*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dX,    N*3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dV,    N*3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dDeg,  N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB,    N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dFX,   N*3*sizeof(float)));

    CUDA_CHECK(cudaMalloc(&dCM,    3*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dSumR2, 1*sizeof(float)));

    CUDA_CHECK(cudaMalloc(&dEdgeU, E*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dEdgeV, E*sizeof(int)));

    reset_sim(dS, dSd, dX, dV, dB, hU, hV, dEdgeU, dEdgeV);

    // CPU scratch for dimension estimate
    std::vector<float> hX(N*3, 0.0f);
    double titleAccum = 0.0;

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        float currentFrame = (float)glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        glfwPollEvents();
        processInput(deltaTime);

        if (keys[GLFW_KEY_R]) {
            reset_sim(dS, dSd, dX, dV, dB, hU, hV, dEdgeU, dEdgeV);
            keys[GLFW_KEY_R] = false;
        }

        // Physics update (multiple substeps if frame time is large)
        if (!paused) {
            int substeps = (int)fminf(8.0f, fmaxf(1.0f, deltaTime / DT));
            for (int s = 0; s < substeps; ++s) {
                compute_degree<<<1, 128>>>(dS, dDeg);
                zero_forces<<<1, 256>>>(dFX);

                edge_forces_and_update_S<<<(E + 255)/256, 256>>>(
                    dS, dSd, dX, dDeg, dB, dEdgeU, dEdgeV, dFX
                );

                reduce_cm_r2<<<1, 256>>>(dX, dCM, dSumR2);

                integrate_X<<<1, 128>>>(dX, dV, dFX, dCM, dSumR2);
            }
        }

        // Map VBOs and fill them directly from CUDA
        CUDA_CHECK(cudaGraphicsMapResources(1, &cudaVboPoints, 0));
        CUDA_CHECK(cudaGraphicsMapResources(1, &cudaVboLines,  0));

        Vertex* dPts = nullptr;
        Vertex* dLines = nullptr;
        size_t ptsBytes = 0, linesBytes = 0;

        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&dPts, &ptsBytes, cudaVboPoints));
        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&dLines, &linesBytes, cudaVboLines));

        fill_points_vbo<<<1, 128>>>(dPts, dX, dDeg, dB);
        fill_lines_vbo<<<(E + 255)/256, 256>>>(dLines, dX, dS, dEdgeU, dEdgeV);

        CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaVboPoints, 0));
        CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaVboLines,  0));

        // Optional: update title with effective dimension estimate (tiny host copy)
        titleAccum += (double)deltaTime;
        if (titleAccum > 0.25) {
            titleAccum = 0.0;
            CUDA_CHECK(cudaMemcpy(hX.data(), dX, N*3*sizeof(float), cudaMemcpyDeviceToHost));
            float deff = effectiveDimensionFromPositions(hX);

            std::ostringstream oss;
            oss.setf(std::ios::fixed);
            oss.precision(2);
            oss << "Zero Player Relational Universe | d_eff ~ " << deff
                << (paused ? " | PAUSED" : "");
            glfwSetWindowTitle(window, oss.str().c_str());
        }

        // Render
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUseProgram(prog);

        glm::mat4 proj = glm::perspective(glm::radians(fov), (float)WIDTH/(float)HEIGHT, 0.05f, 200.0f);
        glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
        glm::mat4 mvp  = proj * view;

        glUniformMatrix4fv(uMVP, 1, GL_FALSE, glm::value_ptr(mvp));

        // Draw lines with additive blend for glow-ish feel
        glBlendFunc(GL_SRC_ALPHA, GL_ONE);
        glUniform1f(uPointSize, 1.0f);

        glBindVertexArray(vaoLines);
        glDrawArrays(GL_LINES, 0, 2 * E);

        // Draw points
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glUniform1f(uPointSize, 7.0f);

        glBindVertexArray(vaoPts);
        glDrawArrays(GL_POINTS, 0, N);

        glBindVertexArray(0);

        glfwSwapBuffers(window);
    }

    // Cleanup
    CUDA_CHECK(cudaGraphicsUnregisterResource(cudaVboPoints));
    CUDA_CHECK(cudaGraphicsUnregisterResource(cudaVboLines));

    glDeleteBuffers(1, &vboPts);
    glDeleteVertexArrays(1, &vaoPts);
    glDeleteBuffers(1, &vboLines);
    glDeleteVertexArrays(1, &vaoLines);
    glDeleteProgram(prog);

    CUDA_CHECK(cudaFree(dS));
    CUDA_CHECK(cudaFree(dSd));
    CUDA_CHECK(cudaFree(dX));
    CUDA_CHECK(cudaFree(dV));
    CUDA_CHECK(cudaFree(dDeg));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dFX));
    CUDA_CHECK(cudaFree(dCM));
    CUDA_CHECK(cudaFree(dSumR2));
    CUDA_CHECK(cudaFree(dEdgeU));
    CUDA_CHECK(cudaFree(dEdgeV));

    glfwTerminate();
    return 0;
}
