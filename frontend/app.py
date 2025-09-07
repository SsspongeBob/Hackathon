import streamlit as st
from datetime import datetime
import base64
import io
from PIL import Image

# 导入机器人代理相关模块
try:
    from langgraph.graph import StateGraph, START, END
    from langgraph.prebuilt.chat_agent_executor import AgentState
    from langgraph.graph.message import add_messages
    from langgraph.store.base import BaseStore
    from langchain_core.runnables import RunnableConfig
    from langchain_core.messages import BaseMessage
    from agent.AgentGraph.utils.image_chain import image_chain
    from typing import TypedDict, Union, Annotated, Sequence
    from robot_function.operator import Operator

    # 初始化操作器
    operator = Operator()
    operator.display()
    operator.init_sam()
    ROBOT_AGENT_AVAILABLE = True
except ImportError as e:
    ROBOT_AGENT_AVAILABLE = False
    print(f"Robot agent modules not available: {e}")

# 设置页面配置
st.set_page_config(
    page_title="URgrasp Chat",
    page_icon="🤖",
    layout="wide"
)

# 机器人状态定义
if ROBOT_AGENT_AVAILABLE:
    class RobotState(AgentState):
        """The state of the robot"""
        question: str
        encoded_image: str
        object_number: int
        place_number: int

    class RobotOutputState(TypedDict):
        """The output state of the robot"""
        object_number: int
        place_number: int
        messages: Annotated[Sequence[BaseMessage], add_messages]

    # 图像识别节点
    def image_recognition_node(
        state: RobotState, config: RunnableConfig, store: BaseStore
    ) -> RobotState:
        res = image_chain.invoke(
            {"question": state["question"],
                "encoded_image": state["encoded_image"]}
        )
        return {"object_number": res["object_number"], "place_number": res["place_number"]}

    # 抓取节点
    def pick_node(state: RobotState):
        operator.pick(state["object_number"])

    # 放置节点
    def place_node(state: RobotState):
        operator.place(state["place_number"])

    # 构建工作流
    graph = StateGraph(state_schema=RobotState, output=RobotOutputState)
    graph.add_node("image_recognition_node", image_recognition_node)
    graph.add_node("pick_node", pick_node)
    graph.add_node("place_node", place_node)
    graph.add_edge(START, "image_recognition_node")
    graph.add_edge("image_recognition_node", "pick_node")
    graph.add_edge("pick_node", "place_node")
    graph.add_edge("place_node", END)

    robot_agent = graph.compile()

# 虚拟模型列表和url (新增更多模型选项)
MODEL_OPTIONS = {
    "URgrasp Robot Agent": "robot://localhost/agent",

}

# 模型对应的图标
MODEL_ICONS = {
    "URgrasp Robot Agent": "✨",
}

# 初始化session state
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "URgrasp Robot Agent" if ROBOT_AGENT_AVAILABLE else "GPT-4"
if "conversations" not in st.session_state:
    st.session_state.conversations = {"新对话": []}
if "current_conv" not in st.session_state:
    st.session_state.current_conv = "新对话"
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

# 辅助函数：处理机器人代理


def process_robot_agent_request(user_input, encoded_image=None):
    """处理机器人代理请求"""
    if not ROBOT_AGENT_AVAILABLE:
        return "❌ 机器人代理模块不可用。请确保已安装所需依赖。"

    try:
        if not encoded_image:
            # 尝试获取操作器的标注图像
            try:
                encoded_image = operator.get_annotated_image()
            except Exception as e:
                return f"❌ 无法获取摄像头图像: {str(e)}"

        results = []
        results.append("🤖 **URgrasp机器人代理处理中...**\n")
        results.append(f"📝 **用户指令**: {user_input}\n")

        # 执行机器人代理流程
        for chunk in robot_agent.stream(
            {
                "question": user_input,
                "encoded_image": encoded_image,
            },
            stream_mode="updates",
        ):
            if chunk:
                for node_name, node_output in chunk.items():
                    if node_name == "image_recognition_node":
                        if "object_number" in node_output and "place_number" in node_output:
                            results.append(f"👁️ **图像识别完成**:")
                            results.append(
                                f"   - 目标物体ID: {node_output['object_number']}")
                            results.append(
                                f"   - 放置位置ID: {node_output['place_number']}\n")

                    elif node_name == "pick_node":
                        results.append("🔧 **执行抓取动作**")
                        results.append(
                            f"   - 抓取物体ID: {node_output.get('object_number', '未知')}\n")

                    elif node_name == "place_node":
                        results.append("📍 **执行放置动作**")
                        results.append(
                            f"   - 放置位置ID: {node_output.get('place_number', '未知')}\n")

        results.append("✅ **任务执行完成！**")
        return "\n".join(results)

    except Exception as e:
        return f"❌ **机器人代理执行错误**: {str(e)}"


def encode_image(image):
    """将PIL图像编码为base64字符串"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


# 自定义CSS样式 - GPT官网风格 (保持原有样式)
st.markdown("""
<style>
/* 全局样式 */
.stApp {
    background-color: #f9f9f9;
}

/* 侧边栏样式 */
section[data-testid="stSidebar"] {
    background-color: #f7f7f8;
    width: 260px !important;
    border-right: 1px solid #e5e5e5;
}

section[data-testid="stSidebar"] > div {
    padding: 0;
    height: 100vh;
}

/* Logo样式 - 黑色并与侧边栏收缩按钮同行 */
.logo-container {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 20px;
    background-color: transparent;
    position: relative;
    height: 56px;
}

.logo {
    color: #000000;
    font-size: 20px;
    font-weight: 600;
    font-family: -apple-system, BlinkMacSystemFont, sans-serif;
    margin-left: 40px;
}

/* 侧边栏内容容器 */
.sidebar-content {
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
}

/* 新建对话按钮 */
.new-chat-btn {
    background-color: transparent;
    border: 1px solid #c5c5d2;
    color: #000000;
    padding: 12px 16px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 14px;
    transition: all 0.15s;
    text-align: center;
    width: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
}

.new-chat-btn:hover {
    background-color: #ececec;
}

/* 历史对话标题 */
.history-title {
    text-align: center;
    color: #666;
    font-size: 14px;
    font-weight: 500;
    margin: 8px 0 4px 0;
}

/* 对话项样式 */
.conversation-item {
    background-color: transparent;
    border: none;
    color: #000000;
    padding: 12px 16px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 14px;
    text-align: left;
    transition: all 0.15s;
    width: 100%;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin-bottom: 2px;
}

.conversation-item:hover {
    background-color: #ececec;
}

.conversation-item.active {
    background-color: #ececec;
    font-weight: 500;
}

/* 主内容区域样式 */
.main-container {
    display: flex;
    flex-direction: column;
    max-width: 800px;
    margin: 0 auto;
    position: relative;
}

.header {
    padding: 16px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    background-color: transparent;
    border: none;
}

/* 聊天容器 - 修复底部间距问题 */
.chat-container {
    flex: 1;
    overflow-y: auto;
    padding: 20px 24px 140px 24px; /* 增加底部padding，为输入框和信息提示留出空间 */
}

.chat-message {
    display: flex;
    padding: 20px 0;
    border-bottom: 1px solid #f0f0f0;
}

.chat-message:last-child {
    border-bottom: none;
}

.message-avatar {
    width: 36px;
    height: 36px;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-right: 16px;
    font-size: 18px;
    flex-shrink: 0;
}

.user-avatar {
    background-color: #10a37f;
    color: white;
}

.assistant-avatar {
    background-color: #6366f1;
    color: white;
}

.message-content {
    flex: 1;
    color: #374151;
    line-height: 1.6;
    font-size: 16px;
    font-family: -apple-system, BlinkMacSystemFont, sans-serif;
}

/* 输入框容器 - 固定在底部，移除白条遮挡 */
.input-container {
    position: fixed;
    bottom: 40px; /* 为信息提示框留出空间 */
    left: 260px;
    right: 0;
    background: transparent;
    padding: 16px 0;
    z-index: 100;
}

.input-wrapper {
    max-width: 800px;
    margin: 0 auto;
    padding: 0 24px;
}

/* 信息提示框样式 - 固定在最底部，恢复原来样式 */
.info-container {
    position: fixed;
    bottom: 0;
    left: 260px;
    right: 0;
    background: white;
    padding: 8px 0;
    z-index: 99;
}

.info-wrapper {
    max-width: 800px;
    margin: 0 auto;
    padding: 0 24px;
}

/* 调整聊天输入框样式 */
.stChatInput {
    background-color: white;
    border: 1px solid #d9d9e3;
    border-radius: 8px;
}

/* 隐藏默认streamlit元素 */
#MainMenu {display: none;}
header[data-testid="stHeader"] {display: none;}
.stDeployButton {display: none;}

/* 调整主内容区域padding */
.main .block-container {
    padding: 0;
    max-width: 100%;
}

/* 侧边栏按钮样式统一 */
section[data-testid="stSidebar"] button {
    width: 100%;
    border: none;
    background-color: transparent;
    color: #000000;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 14px;
    text-align: left;
    margin-bottom: 2px;
    transition: all 0.15s;
}

section[data-testid="stSidebar"] button:hover {
    background-color: #ececec;
}

/* 活跃状态的按钮 */
section[data-testid="stSidebar"] button[data-active="true"] {
    background-color: #ececec;
    font-weight: 500;
}

/* 调整顶部间距 */
section.main > div:first-child {
    padding-top: 0 !important;
}

/* 去除顶部空白 */
.stApp > header {
    display: none;
}

.stApp > div > div {
    padding-top: 0;
}

[data-testid="stAppViewContainer"] {
    padding-top: 0;
}

/* 图像上传区域样式 */
.image-upload-area {
    border: 2px dashed #d9d9e3;
    border-radius: 8px;
    padding: 20px;
    text-align: center;
    margin: 10px 0;
    background-color: #f9f9f9;
}
</style>
""", unsafe_allow_html=True)

# 侧边栏
with st.sidebar:
    # Logo容器
    st.markdown("""
    <div class="logo-container">
        <div class="logo">URgrasp</div>
    </div>
    """, unsafe_allow_html=True)

    # 侧边栏内容
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)

    # 新建对话按钮
    if st.button("➕ 新建对话", key="new_chat", help="创建新对话", use_container_width=True):
        conv_count = len(st.session_state.conversations)
        name = f"对话 {conv_count + 1}"
        st.session_state.conversations[name] = []
        st.session_state.current_conv = name
        st.session_state.uploaded_image = None  # 清除上传的图像
        st.rerun()

    # 历史对话标题
    st.markdown('<div class="history-title">历史对话</div>',
                unsafe_allow_html=True)

    # 历史对话列表
    for conv_name in list(st.session_state.conversations.keys()):
        messages = st.session_state.conversations[conv_name]
        if messages and len(messages) > 0:
            display_name = messages[0]['content'][:20] + "..." if len(
                messages[0]['content']) > 20 else messages[0]['content']
        else:
            display_name = conv_name

        is_active = st.session_state.current_conv == conv_name

        if st.button(
            display_name,
            key=f"conv_{conv_name}",
            help=f"切换到: {conv_name}",
            use_container_width=True
        ):
            st.session_state.current_conv = conv_name
            st.rerun()

    # 机器人状态显示
    if ROBOT_AGENT_AVAILABLE and st.session_state.selected_model == "URgrasp Robot Agent":
        st.markdown("---")
        st.markdown("### 🤖 机器人状态")
        st.success("✅ 机器人代理已连接")
        st.info("📷 摄像头就绪")

        # 图像上传选项（用于测试）
        st.markdown("### 📸 图像输入")
        uploaded_file = st.file_uploader(
            "上传测试图像", type=['png', 'jpg', 'jpeg'], key="image_upload")
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="上传的图像", width=200)
            st.session_state.uploaded_image = encode_image(image)

    st.markdown('</div>', unsafe_allow_html=True)

# 主内容区域
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# 顶部标题和模型选择器
col1, col2, col3 = st.columns([2, 3, 2])
with col2:
    st.markdown("""
    <div class="header">
        <h3 style="margin: 0; color: #374151; text-align: center;">URgrasp Chat</h3>
    </div>
    """, unsafe_allow_html=True)

# 模型选择器
with st.container():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        available_models = list(MODEL_OPTIONS.keys())
        if not ROBOT_AGENT_AVAILABLE:
            available_models = [
                model for model in available_models if model != "URgrasp Robot Agent"]

        selected_model = st.selectbox(
            "选择模型",
            options=available_models,
            index=available_models.index(
                st.session_state.selected_model) if st.session_state.selected_model in available_models else 0,
            key="model_selector",
            label_visibility="collapsed"
        )
        st.session_state.selected_model = selected_model

# 聊天容器
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

messages = st.session_state.conversations[st.session_state.current_conv]

if not messages:
    if selected_model == "URgrasp Robot Agent":
        st.markdown("""
        <div style="text-align: center; padding: 60px 20px; color: #6b7280;">
            <h2 style="color: #374151; margin-bottom: 16px;">🤖 URgrasp 机器人代理</h2>
            <p>我是您的智能机器人助手，可以帮助您进行物体识别、抓取和放置操作！</p>
            <p style="font-size: 14px; margin-top: 20px;">
                <strong>支持的功能：</strong><br>
                • 📷 实时图像识别<br>
                • 🔧 精确物体抓取<br>
                • 📍 智能位置放置<br>
            </p>
            <p style="font-size: 14px; margin-top: 20px; color: #10a37f;"><strong>当前模型: URgrasp Robot Agent</strong></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align: center; padding: 60px 20px; color: #6b7280;">
            <h2 style="color: #374151; margin-bottom: 16px;">👋 欢迎使用 URgrasp Chat</h2>
            <p>开始一个新的对话，我来帮助你解决问题！</p>
            <p style="font-size: 14px; margin-top: 20px;">当前模型: <strong>{}</strong></p>
        </div>
        """.format(selected_model), unsafe_allow_html=True)
else:
    for i, message in enumerate(messages):
        if message['role'] == 'user':
            avatar_class = "user-avatar"
            avatar_icon = "👤"
        else:
            avatar_class = "assistant-avatar"
            # 根据当前选择的模型显示不同的图标
            avatar_icon = MODEL_ICONS.get(st.session_state.selected_model, "🤖")

        st.markdown(f"""
        <div class="chat-message">
            <div class="message-avatar {avatar_class}">
                {avatar_icon}
            </div>
            <div class="message-content">
                {message['content']}
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# 输入框容器 - 固定在底部
st.markdown('<div class="input-container"><div class="input-wrapper">',
            unsafe_allow_html=True)

# 用户输入
user_input = st.chat_input("输入你的消息...")

st.markdown('</div></div>', unsafe_allow_html=True)

# 信息提示容器 - 移到最底部并移除背景
st.markdown('<div class="info-container"><div class="info-wrapper">',
            unsafe_allow_html=True)

# 信息提示
if selected_model == "URgrasp Robot Agent":
    if ROBOT_AGENT_AVAILABLE:
        st.success(f"🤖 **{selected_model}** 已就绪 | 支持实时图像识别和机器人控制")
    else:
        st.error("❌ 机器人代理不可用 | 请检查依赖安装")
else:
    st.info(
        f"当前模型: **{selected_model}** | API端点: `{MODEL_OPTIONS[selected_model]}`")

st.markdown('</div></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# 处理用户输入
if user_input:
    messages = st.session_state.conversations[st.session_state.current_conv]
    messages.append({"role": "user", "content": user_input})

    # 根据选择的模型生成不同的回复
    if selected_model == "URgrasp Robot Agent":
        # 使用机器人代理处理请求
        encoded_image = st.session_state.uploaded_image if st.session_state.uploaded_image else None
        assistant_response = process_robot_agent_request(
            user_input, encoded_image)
    else:
        # 模拟其他AI模型回复
        api_url = MODEL_OPTIONS[selected_model]
        model_icon = MODEL_ICONS.get(selected_model, "🤖")
        assistant_response = f"我是 {selected_model} 模型 {model_icon}。你刚才说：\"{user_input}\"\n\n这是一个模拟回复。实际部署时，这里会调用 {api_url} 来获取真实的AI回复。"

    messages.append({"role": "assistant", "content": assistant_response})
    st.rerun()
