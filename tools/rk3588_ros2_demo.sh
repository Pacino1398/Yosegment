#!/bin/bash
# Yosegment ROS2 Octomap Demo 脚本 (RK3588 Ubuntu)
# 用途：在 RK3588 上订阅并查看 Yosegment 发布的 Octomap 地图数据

set -e

# ==================== 配置区 ====================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YOSEGMENT_DIR="${SCRIPT_DIR}/.."

# ROS2 环境设置（根据你的安装路径调整）
ROS2_DISTRO="humble"  # 或 foxy, galactic, iron 等
ROS2_SETUP="/opt/ros/${ROS2_DISTRO}/setup.bash"

# 话题配置
OCTOMAP_TOPIC="/yoseg/octomap"
OCC2D_TOPIC="/yoseg/occ2d_grid"
ZBAND_TOPIC="/yoseg/z_band_markers"
SNAPSHOT_TOPIC="/yoseg/octomap_snapshot_json"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ==================== 函数定义 ====================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查环境
check_environment() {
    log_info "检查环境..."
    
    # 检查 ROS2 环境
    if [ ! -f "$ROS2_SETUP" ]; then
        log_error "未找到 ROS2 环境：$ROS2_SETUP"
        log_info "请确认 ROS2 ${ROS2_DISTRO} 已安装，或修改脚本中的 ROS2_DISTRO 变量"
        return 1
    fi
    
    # 检查 Python
    if ! command -v python3 &> /dev/null; then
        log_error "未找到 python3"
        return 1
    fi
    
    # 检查 ros2 命令
    source "$ROS2_SETUP"
    if ! command -v ros2 &> /dev/null; then
        log_error "未找到 ros2 命令"
        return 1
    fi
    
    log_success "环境检查通过"
    return 0
}

# 检查依赖包
check_dependencies() {
    log_info "检查 Python 依赖..."
    
    # 检查 octomap
    if ! python3 -c "import octomap" 2>/dev/null; then
        log_warn "未安装 python-octomap，尝试安装..."
        pip3 install octomap || log_error "安装 octomap 失败"
    fi
    
    log_success "依赖检查完成"
}

# 显示帮助
show_help() {
    cat << EOF
Yosegment ROS2 Octomap Demo - RK3588 操作指南
============================================

用法：
  $(basename $0) [选项]

选项:
  -h, --help          显示帮助信息
  -c, --check         仅检查环境
  -p, --publish       发布示例 Octomap 数据
  -s, --subscribe     订阅 Octomap 话题
  -l, --list          列出所有 yoseg 相关话题
  -r, --rviz          启动 RViz2 查看地图
  -a, --all           一键演示（发布 + 订阅 + RViz）
  --npz <path>        指定 NPZ 文件路径（默认使用示例数据）

示例:
  # 1. 检查环境
  $(basename $0) --check
  
  # 2. 发布示例数据
  $(basename $0) --publish
  
  # 3. 订阅话题查看数据
  $(basename $0) --subscribe
  
  # 4. 启动 RViz2 可视化
  $(basename $0) --rviz
  
  # 5. 一键完整演示
  $(basename $0) --all

环境要求:
  - Ubuntu 20.04/22.04 (RK3588)
  - ROS2 Humble (或兼容版本)
  - Python 3.8+
  - python-octomap

注意事项:
  - 首次运行前请确保已 source ROS2 环境
  - RViz2 需要显示环境支持
  - RK3588 建议使用轻量级查看方式

EOF
}

# 发布示例数据
publish_demo() {
    log_info "启动 Octomap 发布器..."
    
    # 检查是否有示例 NPZ 文件
    SAMPLE_NPZ="${YOSEGMENT_DIR}/data/sample_map.npz"
    if [ ! -f "$SAMPLE_NPZ" ]; then
        log_warn "未找到示例 NPZ 文件，尝试生成..."
        python3 "${YOSEGMENT_DIR}/tools/generate_sample_map.py" || {
            log_error "生成示例文件失败"
            return 1
        }
    fi
    
    source "$ROS2_SETUP"
    python3 "${YOSEGMENT_DIR}/app/ros2/occ_zband_publisher.py" \
        --npz "$SAMPLE_NPZ" \
        --frame-id map \
        --resolution 1.0 \
        --rate 2.0 \
        --publish-octomap
    
    log_success "发布器已启动"
}

# 订阅话题
subscribe_demo() {
    log_info "订阅话题：$OCTOMAP_TOPIC"
    
    source "$ROS2_SETUP"
    
    # 使用 ros2 topic echo 查看消息
    ros2 topic echo "$OCTOMAP_TOPIC" --once || {
        log_warn "无法获取消息，确认发布者正在运行"
        log_info "使用 ros2 topic list 查看可用话题"
        ros2 topic list | grep yoseg
    }
}

# 列出话题
list_topics() {
    log_info "Yosegment 相关话题:"
    
    source "$ROS2_SETUP"
    ros2 topic list | grep -E "(yoseg|octomap)" || {
        log_warn "未找到相关话题"
    }
    
    echo ""
    log_info "话题详情:"
    ros2 topic info "$OCTOMAP_TOPIC" 2>/dev/null || log_warn "话题 $OCTOMAP_TOPIC 不存在"
    ros2 topic info "$OCC2D_TOPIC" 2>/dev/null || log_warn "话题 $OCC2D_TOPIC 不存在"
    ros2 topic info "$ZBAND_TOPIC" 2>/dev/null || log_warn "话题 $ZBAND_TOPIC 不存在"
}

# 启动 RViz2
start_rviz() {
    log_info "启动 RViz2..."
    
    source "$ROS2_SETUP"
    
    # 创建 RViz 配置文件
    RVIZ_CONFIG="/tmp/yosegment_rviz_config.rviz"
    cat > "$RVIZ_CONFIG" << 'RVIZ_EOF'
Panels:
  - Class: rviz_common/Displays
    Name: Displays
Visualization Manager:
  Displays:
    - Alpha: 0.5
      Class: rviz_default_plugins/Map
      Enabled: true
      Name: Map
      Topic:
        Value: /yoseg/occ2d_grid
    - Class: rviz_default_plugins/MarkerArray
      Enabled: true
      Name: Markers
      Topic:
        Value: /yoseg/z_band_markers
  Global Options:
    Background Color: 48, 48, 48
    Fixed Frame: map
  Tools:
    - Class: rviz_default_plugins/MoveCamera
RVIZ_EOF

    rviz2 -d "$RVIZ_CONFIG" &
    log_success "RViz2 已启动"
}

# 一键演示
run_all_demo() {
    log_info "启动完整演示..."
    
    source "$ROS2_SETUP"
    
    # 后台启动发布者
    log_info "启动发布者（后台运行）..."
    python3 "${YOSEGMENT_DIR}/app/ros2/occ_zband_publisher.py" \
        --npz "${YOSEGMENT_DIR}/data/sample_map.npz" \
        --frame-id map \
        --resolution 1.0 \
        --rate 2.0 \
        --publish-octomap &
    PUB_PID=$!
    
    sleep 2
    
    # 显示话题列表
    log_info "当前 ROS2 话题:"
    ros2 topic list | grep yoseg
    
    # 等待用户
    echo ""
    log_info "发布者正在运行 (PID: $PUB_PID)"
    log_info "按 Ctrl+C 停止"
    
    wait $PUB_PID
}

# ==================== 主程序 ====================

main() {
    if [ $# -eq 0 ]; then
        show_help
        exit 0
    fi
    
    while [ $# -gt 0 ]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -c|--check)
                check_environment
                check_dependencies
                exit 0
                ;;
            -p|--publish)
                check_environment || exit 1
                publish_demo
                exit 0
                ;;
            -s|--subscribe)
                check_environment || exit 1
                subscribe_demo
                exit 0
                ;;
            -l|--list)
                check_environment || exit 1
                list_topics
                exit 0
                ;;
            -r|--rviz)
                check_environment || exit 1
                start_rviz
                exit 0
                ;;
            -a|--all)
                check_environment || exit 1
                run_all_demo
                exit 0
                ;;
            --npz)
                shift
                NPZ_PATH="$1"
                ;;
            *)
                log_error "未知选项：$1"
                show_help
                exit 1
                ;;
        esac
        shift
    done
}

main "$@"