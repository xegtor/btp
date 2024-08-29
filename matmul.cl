////////////////////////////////////////////////////////////////
// L3_SLM_Fast_16x8_2

#define VEC_SIZE        4
#define TILE_M          64
#define TILE_K          64
#define TILE_N          128
#define ROWS_PER_WI     8

__attribute__((reqd_work_group_size(16, 8, 1)))
__kernel void L3_SLM_8x8_8x16(
    const __global float4 *src0,
    const __global float4 *src1,
    __global float4 *dst,const float alpha,
    const float beta,
    int width0,
    int width1)
{
    width0 /= VEC_SIZE;
    width1 /= VEC_SIZE;

    // Src0 atile is M rows x K columns
    // M = 64
    // K = 64 = 16 float4s
    // requires sizeof(float) x M x K = 16K SLM
    __local float4 atile[TILE_M * TILE_K / VEC_SIZE];

    const int group_x = get_group_id(0);
    const int group_y = get_group_id(1);
    const int local_x = get_local_id(0);
    const int local_y = get_local_id(1);

    // Result ctile is M rows x N columns
    // M = 64, we have 8 rows of work-items, so we need 64/8 8 results down
    // N = 128, we have 16 columns of work-items, so we need 128/16 = 8 results across = 2 float4s across

    float4 dot00 = (float4)(0.f);
    float4 dot01 = (float4)(0.f);
    float4 dot02 = (float4)(0.f);
    float4 dot03 = (float4)(0.f);
    float4 dot04 = (float4)(0.f);
    float4 dot05 = (float4)(0.f);
    float4 dot06 = (float4)(0.f);
    float4 dot07 = (float4)(0.f);

    float4 dot10 = (float4)(0.f);
    float4 dot11 = (float4)(0.f);
    float4 dot12 = (float4)(0.f);
    float4 dot13 = (float4)(0.f);
    float4 dot14 = (float4)(0.f);
    float4 dot15 = (float4)(0.f);
    float4 dot16 = (float4)(0.f);
    float4 dot17 = (float4)(0.f);

    __global float4 *dst_write0 = dst + local_x + ( group_x * ( TILE_N / VEC_SIZE ) ) + ( ( group_y * TILE_M ) + ROWS_PER_WI * local_y ) * width1;

    // Src0 is used to load atile.
    // It starts at the left side of src0 and walks across.
    // atile is M rows x K columns.
    const __global float4 *src0_read = src0 + local_x + ( ( group_y * TILE_M ) + ROWS_PER_WI * local_y ) * width0;

    // Src1 is directly used as btile.
    // It starts at the top of src1 and walks down.
    // btile is K rows x N columns.
    // K = 64, we'll process four rows at a time
    // N = 128, we have 16 columns of work-items, so we need 128/16 = 8 floats across = 2 float4s across
    const __global float4 *src1_read0 = src1 + local_x + ( group_x * ( TILE_N / VEC_SIZE ) );
    const __global float4 *src1_read1 = src1_read0 + ( TILE_N / 2 / VEC_SIZE );
  
    __local float4 *slm = atile + local_y * ( ROWS_PER_WI * TILE_K / VEC_SIZE );

    // Walk ACROSS src0 and DOWN src1:
    int w = 0;
    do
    {
        // We want to load atile, which is M rows x K columns
        // M = 64, and we have 8 rows of work-items, so each work-item must load 64/8 = 8 rows.
        // K = 64, and we have 16 columns of work-items, so each work-item must load 64/16 = 4 columns = 1 float4.
        slm[local_x + 0 * TILE_K / VEC_SIZE] = src0_read[0 * width0];
        slm[local_x + 1 * TILE_K / VEC_SIZE] = src0_read[1 * width0];
        slm[local_x + 2 * TILE_K / VEC_SIZE] = src0_read[2 * width0];
        slm[local_x + 3 * TILE_K / VEC_SIZE] = src0_read[3 * width0];
        slm[local_x + 4 * TILE_K / VEC_SIZE] = src0_read[4 * width0];
        slm[local_x + 5 * TILE_K / VEC_SIZE] = src0_read[5 * width0];
        slm[local_x + 6 * TILE_K / VEC_SIZE] = src0_read[6 * width0];
        slm[local_x + 7 * TILE_K / VEC_SIZE] = src0_read[7 * width0];
        src0_read += TILE_K / VEC_SIZE;

        barrier(CLK_LOCAL_MEM_FENCE);

        int i = 0;
        do
        {
            // We get better performance by loading btile first.
            const float4 brow00 = src1_read0[0];   src1_read0 += width1;
            const float4 brow01 = src1_read0[0];   src1_read0 += width1;
            const float4 brow02 = src1_read0[0];   src1_read0 += width1;
            const float4 brow03 = src1_read0[0];   src1_read0 += width1;
            const float4 brow10 = src1_read1[0];   src1_read1 += width1;
            const float4 brow11 = src1_read1[0];   src1_read1 += width1;
            const float4 brow12 = src1_read1[0];   src1_read1 += width1;
            const float4 brow13 = src1_read1[0];   src1_read1 += width1;

            const float4 a0 = slm[ i + 0 * TILE_K / VEC_SIZE ];
            dot00 = mad(brow00, (float4) a0.x, dot00);
            dot00 = mad(brow01, (float4) a0.y, dot00);
            dot00 = mad(brow02, (float4) a0.z, dot00);
            dot00 = mad(brow03, (float4) a0.w, dot00);
            dot10 = mad(brow10, (float4) a0.x, dot10);
            dot10 = mad(brow11, (float4) a0.y, dot10);
            dot10 = mad(brow12, (float4) a0.z, dot10);
            dot10 = mad(brow13, (float4) a0.w, dot10);

            const float4 a1 = slm[ i + 1 * TILE_K / VEC_SIZE ];
            dot01 = mad(brow00, (float4) a1.x, dot01);
            dot01 = mad(brow01, (float4) a1.y, dot01);
            dot01 = mad(brow02, (float4) a1.z, dot01);
            dot01 = mad(brow03, (float4) a1.w, dot01);
            dot11 = mad(brow10, (float4) a1.x, dot11);
            dot11 = mad(brow11, (float4) a1.y, dot11);
            dot11 = mad(brow12, (float4) a1.z, dot11);
            dot11 = mad(brow13, (float4) a1.w, dot11);

            const float4 a2 = slm[ i + 2 * TILE_K / VEC_SIZE ];
            dot02 = mad(brow00, (float4) a2.x, dot02);
            dot02 = mad(brow01, (float4) a2.y, dot02);
            dot02 = mad(brow02, (float4) a2.z, dot02);
            dot02 = mad(brow03, (float4) a2.w, dot02);
            dot12 = mad(brow10, (float4) a2.x, dot12);
            dot12 = mad(brow11, (float4) a2.y, dot12);
            dot12 = mad(brow12, (float4) a2.z, dot12);
            dot12 = mad(brow13, (float4) a2.w, dot12);

            const float4 a3 = slm[ i + 3 * TILE_K / VEC_SIZE ];
            dot03 = mad(brow00, (float4) a3.x, dot03);
            dot03 = mad(brow01, (float4) a3.y, dot03);
            dot03 = mad(brow02, (float4) a3.z, dot03);
            dot03 = mad(brow03, (float4) a3.w, dot03);
            dot13 = mad(brow10, (float4) a3.x, dot13);
            dot13 = mad(brow11, (float4) a3.y, dot13);
            dot13 = mad(brow12, (float4) a3.z, dot13);
            dot13 = mad(brow13, (float4) a3.w, dot13);

            const float4 a4 = slm[ i + 4 * TILE_K / VEC_SIZE ];
            dot04 = mad(brow00, (float4) a4.x, dot04);
            dot04 = mad(brow01, (float4) a4.y, dot04);
            dot04 = mad(brow02, (float4) a4.z, dot04);
            dot04 = mad(brow03, (float4) a4.w, dot04);
            dot14 = mad(brow10, (float4) a4.x, dot14);
            dot14 = mad(brow11, (float4) a4.y, dot14);
            dot14 = mad(brow12, (float4) a4.z, dot14);
            dot14 = mad(brow13, (float4) a4.w, dot14);

            const float4 a5 = slm[ i + 5 * TILE_K / VEC_SIZE ];
            dot05 = mad(brow00, (float4) a5.x, dot05);
            dot05 = mad(brow01, (float4) a5.y, dot05);
            dot05 = mad(brow02, (float4) a5.z, dot05);
            dot05 = mad(brow03, (float4) a5.w, dot05);
            dot15 = mad(brow10, (float4) a5.x, dot15);
            dot15 = mad(brow11, (float4) a5.y, dot15);
            dot15 = mad(brow12, (float4) a5.z, dot15);
            dot15 = mad(brow13, (float4) a5.w, dot15);

            const float4 a6 = slm[ i + 6 * TILE_K / VEC_SIZE ];
            dot06 = mad(brow00, (float4) a6.x, dot06);
            dot06 = mad(brow01, (float4) a6.y, dot06);
            dot06 = mad(brow02, (float4) a6.z, dot06);
            dot06 = mad(brow03, (float4) a6.w, dot06);
            dot16 = mad(brow10, (float4) a6.x, dot16);
            dot16 = mad(brow11, (float4) a6.y, dot16);
            dot16 = mad(brow12, (float4) a6.z, dot16);
            dot16 = mad(brow13, (float4) a6.w, dot16);

            const float4 a7 = slm[ i + 7 * TILE_K / VEC_SIZE ];
            dot07 = mad(brow00, (float4) a7.x, dot07);
            dot07 = mad(brow01, (float4) a7.y, dot07);
            dot07 = mad(brow02, (float4) a7.z, dot07);
            dot07 = mad(brow03, (float4) a7.w, dot07);
            dot17 = mad(brow10, (float4) a7.x, dot17);
            dot17 = mad(brow11, (float4) a7.y, dot17);
            dot17 = mad(brow12, (float4) a7.z, dot17);
            dot17 = mad(brow13, (float4) a7.w, dot17);

            i++;
        }
        while( i < TILE_K / VEC_SIZE );

        barrier(CLK_LOCAL_MEM_FENCE);

        w += TILE_K / VEC_SIZE;
    }
    while( w < width0 );

    __global float4 *dst_write1 = dst_write0 + ( TILE_N / 2 / VEC_SIZE );

    dst_write0[0] = dot00;  dst_write0 += width1;
    dst_write0[0] = dot01;  dst_write0 += width1;
    dst_write0[0] = dot02;  dst_write0 += width1;
    dst_write0[0] = dot03;  dst_write0 += width1;
    dst_write0[0] = dot04;  dst_write0 += width1;
    dst_write0[0] = dot05;  dst_write0 += width1;
    dst_write0[0] = dot06;  dst_write0 += width1;
    dst_write0[0] = dot07;  dst_write0 += width1;

    dst_write1[0] = dot10;  dst_write1 += width1;
    dst_write1[0] = dot11;  dst_write1 += width1;
    dst_write1[0] = dot12;  dst_write1 += width1;
    dst_write1[0] = dot13;  dst_write1 += width1;
    dst_write1[0] = dot14;  dst_write1 += width1;
    dst_write1[0] = dot15;  dst_write1 += width1;
    dst_write1[0] = dot16;  dst_write1 += width1;
    dst_write1[0] = dot17;  dst_write1 += width1;
}

#undef VEC_SIZE
#undef TILE_M
#undef TILE_K
#undef TILE_N
#undef ROWS_PER_WI
