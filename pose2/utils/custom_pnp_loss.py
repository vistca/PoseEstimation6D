# from kornia.geometry.calibration.pnp import solve_pnp_dlt

# def custom_loss(pred_points, nr_points):


#     camera_matrix = np.array([
#             [572.4114, 0.0, 325.2611],
#             [0.0, 573.57043, 242.04899],
#             [0.0, 0.0, 1.0]
#         ], dtype=np.float32)

#     result = torch.empty(nr_datapoints, 12, dtype=torch.float32)

#     for i in range(nr_datapoints):
#         pred = preds[i]
#         pred_points = pred.squeeze(0)

#         model_points_2d = np.empty((8,2))
        
#         nr_points = pred_points.size()[0] // 2


#         for j in range(nr_points):
#             point_x = pred_points[2*j].item() 
#             point_y = pred_points[2*j+1].item() 
#             model_points_2d[j] = point_x, point_y

#         model_points_3d = models_points_3d[i]

#         success, pred_rvec, pred_pos = cv2.solvePnP(
#             objectPoints=model_points_3d.numpy(),
#             imagePoints=model_points_2d,
#             cameraMatrix=camera_matrix,
#             distCoeffs=np.zeros(5),
#             flags=cv2.SOLVEPNP_ITERATIVE
#         )

#         pred_rot_matrix, _ = cv2.Rodrigues(pred_rvec)
#         pred_t = torch.tensor(pred_pos, dtype=torch.float32).squeeze()
#         pred_R = torch.tensor(pred_rot_matrix, dtype=torch.float32).flatten()

#         result[i] = torch.cat((pred_t, pred_R), dim=0)

#     solve_pnp_dlt()