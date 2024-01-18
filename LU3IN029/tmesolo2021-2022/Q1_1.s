.data

user_num:	.word	0
tab_chiffre:	.space	40

.text	
main:
# prologue
	addiu	$29,$29,-4 # variable locale
	sw	$0,0($29)
# scanf("%d", &user_num);
	ori	$2,$0,5
	syscall
	ori	$4,$2,0
	lui	$8,0x1001
	sw	$4,0($8)
	addiu	$8,$8,4
	ori	$5,$8,0
	jal	occ_num_chiffre
	sw	$2,0($29)
	ori	$4,$2,0
	ori	$2,$0,1
	syscall
	
# epilogue 
	addiu	$29,$29,4
#exit
	ori	$2,$0,10
	syscall



occ_num_chiffre:
# prologue
	addiu	$29,$29,-16 #3 variable local + r31
	sw	$31,12($29)
	ori	$2,$0,1 # nb_c=1
	sw	$4,4($29) # q =n
	ori	$8,$0,0 # q
	
	ori	$10,$0,10 # r10 =10
	
while:
# q >= 10
	sub	$9,$4,$10
	bltz	$9, endwhile
	
	div	$4,$10
	mfhi	$8 # r = q % 10;
	mflo	$4 # q = q / 10;
	
# t[r] += 1;
	ori	$9,$0,4
	mult	$8,$9
	mflo	$9
	addu	$5,$5,$9
	lw	$11,0($5)
	addiu	$11,$11,1
	sw	$11,0($5)
	sub	$5,$5,$9

# nb_c += 1; 
	addiu	$2,$2,1
	j while

endwhile:
# t[q] += 1;
	ori	$9,$0,4
	mult	$4,$9
	mflo	$9
	addu	$5,$5,$9
	lw	$11,0($5)
	addiu	$11,$11,1
	sw	$11,0($5)
	sub	$5,$5,$9
	
#epilogue
	lw	$31,12($29)
	addiu	$29,$29,16
	jr	$31
	

	
	
	