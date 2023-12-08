import argparse

import onnx
import onnx_graphsurgeon as gs
import onnxsim
import numpy as np
from onnx import shape_inference
from collections import OrderedDict

def EffcientNMS_TRT(input_path, output_path, class_num, score_threshold = 0.25, iou_threshold = 0.45, max_boxes = 100, mul_name = "p2o.Mul.156", concat_name = "p2o.Concat.28"):
    #注意修改
    ########################################################
    INPUT_PATH = input_path
    WEIGHTS_TYPE = "s"
    SAVE_PATH = output_path
    CLASS_NUM = class_num
    SCORE_THRESHOLD = score_threshold
    IOU_THRESHOLD = iou_threshold
    MAX_BOXES = max_boxes
    ########################################################

    # if(WEIGHTS_TYPE=="s"):
    #     Mul_name = 'Mul_78'
    # elif(WEIGHTS_TYPE=="m"):
    #     Mul_name = 'Mul_100'
    # elif(WEIGHTS_TYPE=="l"):
    #     Mul_name = 'Mul_122'
    # elif(WEIGHTS_TYPE=="x"):
    #     Mul_name = 'Mul_144'
    Mul_name = mul_name
    Concat_name = concat_name

    gs_graph = gs.import_onnx(onnx.load(INPUT_PATH))
    # fold constants
    gs_graph.fold_constants()
    gs_graph.cleanup().toposort()

    Mul = [node for node in gs_graph.nodes if node.name==Mul_name][0]
    Concat_14 = [node for node in gs_graph.nodes if node.name==Concat_name][0]

    scores = gs.Variable(name='scores',shape=[1,8400,CLASS_NUM],dtype=np.float32)
    Transpose = gs.Node(name='lastTranspose',op='Transpose',
                    inputs=[Concat_14.outputs[0]],
                    outputs=[scores],
                    attrs=OrderedDict(perm=[0,2,1]))
    gs_graph.nodes.append(Transpose)

    Mul.outputs[0].name = 'boxes'
    gs_graph.inputs = [gs_graph.inputs[0]] # 去掉input中原来包含的scale_factor输入
    gs_graph.outputs = [Mul.outputs[0],scores]
    gs_graph.outputs[0].dtype=np.float32
    gs_graph.outputs[1].dtype=np.float32

    gs_graph.cleanup().toposort()
    onnx_graph = shape_inference.infer_shapes(gs.export_onnx(gs_graph))
    onnx_graph, check = onnxsim.simplify(onnx_graph)

    gs_graph = gs.import_onnx(onnx_graph)
    op_inputs = gs_graph.outputs
    op = "EfficientNMS_TRT"
    attrs = {
        "plugin_version": "1",
        "background_class": -1,
        "max_output_boxes": MAX_BOXES,
        "score_threshold": SCORE_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
        "score_activation": False,
        "box_coding": 0,
    }

    output_num_detections = gs.Variable(
        name="num_dets",
        dtype=np.int32,
        shape=[1, 1],
    )
    output_boxes = gs.Variable(
        name="det_boxes",
        dtype=np.float32,
        shape=[1, 100, 4],
    )
    output_scores = gs.Variable(
        name="det_scores",
        dtype=np.float32,
        shape=[1, 100],
    )
    output_labels = gs.Variable(
        name="det_classes",
        dtype=np.int32,
        shape=[1, 100],
    )
    op_outputs = [
        output_num_detections, output_boxes, output_scores, output_labels
    ]

    TRT = gs.Node(op=op,name="batched_nms",inputs=op_inputs,outputs=op_outputs,attrs=attrs)
    gs_graph.nodes.append(TRT)
    gs_graph.outputs = op_outputs
    gs_graph.cleanup().toposort()

    onnx.save(gs.export_onnx(gs_graph),SAVE_PATH)
    print("finished")

def main(args):
    # builder.create_network(args.onnx, args.end2end, args.conf_thres, args.iou_thres, args.max_det, v8=args.v8)
    # builder.create_engine(args.engine, args.precision, args.calib_input, args.calib_cache, args.calib_num_images,
    #                       args.calib_batch_size)
    EffcientNMS_TRT(args.input, args.output, args.class_num, args.score_thres, args.iou_thres, args.max_boxes, args.mul_layer, args.concat_layer)

# def test():
#     EffcientNMS_TRT("models/ppyoloe_crn_s_36e_pphuman/model.onnx",
#                     "models/ppyoloe_crn_s_36e_pphuman/model_w_nmsrt.test.onnx",
#                     1)

if __name__ == "__main__":
    # test()
    # exit(0)
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="The input ONNX model file to load")
    parser.add_argument("-o", "--output", required=True, help="The output path for the ONNX model")
    # parser.add_argument("-p", "--precision", default="fp16", choices=["fp32", "fp16", "int8"],
                        # help="The precision mode to build in, either 'fp32', 'fp16' or 'int8', default: 'fp16'")
    # parser.add_argument("-v", "--verbose", action="store_true", help="Enable more verbose log output")
    # parser.add_argument("-w", "--workspace", default=1, type=int, help="The max memory workspace size to allow in Gb, "
    #                                                                    "default: 1")
    # parser.add_argument("--calib_input", help="The directory holding images to use for calibration")
    # parser.add_argument("--calib_cache", default="./calibration.cache",
                        # help="The file path for INT8 calibration cache to use, default: ./calibration.cache")
    # parser.add_argument("--calib_num_images", default=5000, type=int,
                        # help="The maximum number of images to use for calibration, default: 5000")
    # parser.add_argument("--calib_batch_size", default=8, type=int,
                        # help="The batch size for the calibration process, default: 8")
    # parser.add_argument("--end2end", default=False, action="store_true",
                        # help="export the engine include nms plugin, default: False")
    parser.add_argument("--score_thres", default=0.25, type=float,
                        help="The conf threshold for the nms, default: 0.25")
    parser.add_argument("--iou_thres", default=0.45, type=float,
                        help="The iou threshold for the nms, default: 0.45")
    parser.add_argument("--max_boxes", default=100, type=int,
                        help="The total num for results, default: 100")
    # parser.add_argument("--v8", default=False, action="store_true",
                        # help="use yolov8 model, default: False")
    parser.add_argument("-c", "--class_num", default=1, help="Num of classes.")
    parser.add_argument("--mul_layer", default="p2o.Mul.156", help="Mul op layer name, export of boxes")
    parser.add_argument("--concat_layer", default="p2o.Concat.28", help="Mul op layer name, export of classes")
    args = parser.parse_args()
    print(args)
    # if not all([args.onnx, args.engine]):
    #     parser.print_help()
    #     # log.error("These arguments are required: --onnx and --engine")
    #     sys.exit(1)
    # if args.precision == "int8" and not (args.calib_input or os.path.exists(args.calib_cache)):
    #     parser.print_help()
    #     # log.error("When building in int8 precision, --calib_input or an existing --calib_cache file is required")
    #     sys.exit(1)
    
    main(args)


