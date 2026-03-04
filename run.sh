### Training ###
# python -m mobileposer.train \
#     --dataset lingo

# python -m mobileposer.train \
#     --dataset all


### Merge multiple checkpoints ###
# python -m mobileposer.combine_weights \
#     --checkpoint all_MotionGV_no_noise

# python -m mobileposer.combine_weights \
#     --checkpoint all_MotionGV_all_noise


### Evaluation ###
# python -m mobileposer.evaluate \
#     --model checkpoints/2/base_model.pth \
#     --dataset lingo



# # LINGO / global (5-pt)
# python -m mobileposer.evaluate \
#     --model checkpoints/all_MotionGV_no_noise/base_model.pth \
#     --dataset lingo \
#     --combo global

# # LINGO / lw_rp_h (3-pt)
# python -m mobileposer.evaluate \
#     --model checkpoints/all_MotionGV_no_noise/base_model.pth \
#     --dataset lingo \
#     --combo lw_rp_h

# # HumanML / global (5-pt)
# python -m mobileposer.evaluate \
#     --model checkpoints/all_MotionGV_no_noise/base_model.pth \
#     --dataset humanml \
#     --combo global

# # HumanML / lw_rp_h (3-pt)
# python -m mobileposer.evaluate \
#     --model checkpoints/all_MotionGV_no_noise/base_model.pth \
#     --dataset humanml \
#     --combo lw_rp_h



# # ParaHome / global (5-pt)
# python -m mobileposer.evaluate \
#     --model checkpoints/all_MotionGV_no_noise/base_model.pth \
#     --dataset parahome \
#     --combo global

# # Humoto / global (5-pt)
# python -m mobileposer.evaluate \
#     --model checkpoints/all_MotionGV_no_noise/base_model.pth \
#     --dataset humoto \
#     --combo global



python -m mobileposer.train \
    --dataset "LINGO,humanml" \
    --body-model smplx