import mujoco as mj
import time
import numpy as np
import cv2



def main():
    max_width = 100
    max_height = 100
    #ctx = mj.GLContext(max_width, max_height)
    #ctx.make_current()

    cam = mj.MjvCamera()
    opt = mj.MjvOption()

    mj.glfw.glfw.init()
    window = mj.glfw.glfw.create_window(640, 480, "Demo", None, None)
    mj.glfw.glfw.make_context_current(window)
    mj.glfw.glfw.swap_interval(1)

    mj.mjv_defaultCamera(cam)
    mj.mjv_defaultOption(opt)

    xml_path = "../agility_cassie/scene.xml"
    model = mj.MjModel.from_xml_path(xml_path)
    data = mj.MjData(model)

    scene = mj.MjvScene(model, maxgeom=10000)
    context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

    while not mj.glfw.glfw.window_should_close(window):
        simstart = data.time

        while (data.time - simstart < 1.0/60.0):
            mj.mj_step(model, data)

        #viewport = mj.MjrRect(0, 0, 0, 0)
        #mj.glfw.glfw.get_framebuffer_size(window)
        viewport = mj.MjrRect(0, 0, 640, 480)

        # This code is for rendering the image in the viewer.
        mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
        mj.mjr_render(viewport, scene, context)

        mj.glfw.glfw.swap_buffers(window)
        mj.glfw.glfw.poll_events()

        # This code is for getting image as numpy array.
        # based on test_rending example https://github.com/deepmind/mujoco/blob/cce35e1816887b34d9c0e96d90d17a14169719d6/python/mujoco/render_test.py#L67 
        upside_down_image = np.empty((480,640, 3), dtype=np.uint8)
        mj.mjr_readPixels(upside_down_image, None, viewport, context)

        # Doing some processing using cv2
        rotated180 = cv2.flip(upside_down_image, flipCode=0)

        # changing the rgb -> bgr because opencv uses bgr
        bgr_image = cv2.cvtColor(rotated180, cv2.COLOR_RGB2BGR)

        cv2.imshow("opencv image", bgr_image)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break


    cv2.destroyAllWindows()
    mj.glfw.glfw.terminate()


if __name__ == "__main__":
    main()

