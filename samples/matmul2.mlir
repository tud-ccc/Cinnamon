func.func @main2() {
	%a = bufferization.alloc_tensor() : tensor<1024x1024xf32>
	%b = bufferization.alloc_tensor() : tensor<1024x1024xf32>
	%c = bufferization.alloc_tensor() : tensor<1024x1024xf32>
	affine.for %i0 = 0 to 16 {						// dims from device architecture
		affine.for %i1 = 0 to 16 {					// dims from device architecture
			affine.for %i2 = 0 to 32 {				// dims from device architecture
				affine.for %i3 = 0 to 8 {			// dims from device architecture
					affine.for %i4 = 0 to 4 {		// dims from device architecture
						affine.for %i5 = 0 to 4 {	// remainder to get to 1024 * 1024
							%id = ((((%i0 * 16 + %i1) * 32 + %i2) * 8 + %i3) * 4 + %i5) * 4 + %i5
							%x = %id % 1024
							%y = $id / 1024
							%acc = affine.for %i = 0 to 1024 {
								yield %acc + %a[%i, %y] * %b[%x, %i]
							}
							%c[x, y] = %acc
						}
					}
				}
			}
		}
	}
	return
}
