.data



.text






moyenne3:
#prologue
	addiu	$29,$29,-12
	sw	$31,8($29)
	sw	
#	int sum = p + q + r;
	addiu	$8,$4,$5
	addiu	$16,$8,$6

# return sum / 3;
	ori	$9,$0,3	
	div	$16,$9
	mflo	$2

# epilogue
	lw	$16,4($29)
	lw	$31,8($29)
	
	addiu	$29,$29,12
	jr	$31	
	
	