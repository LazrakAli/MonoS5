.data

tab:	.word	23,7,12,513,-1

.text

main:
# prologue
	addiu	$29,$29,-8
# int x = arimean(tab);
	lui	$4,0x1001
	sw	$4,4($29)
	jal	arimean
	sw	$2,0($29)
# printf("%d", x);
	ori	$4,$2,0
	ori	$2,$0,1
	syscall
# epilogue
	addiu	$29,$29,8
# exit()
	ori	$2,$0,10
	syscall
	
arimean:
# prologue
	addiu	$29,$29,-16
	sw	$31,12($29)
	sw	$4,0($29)
# appel de sizetab(n)
	jal	sizetab
	sw	$2,4($29)
	lw	$4,0($29)
# appel de sumtab
	jal	sumtab
	sw	$2,8($29)
	lw	$8,4($29)
	div	$2,$8
	mflo	$2
# epilogue
	lw	$31,12($29)
	addiu	$29,$29,16
	jr	$31
	
sizetab:
# prologue
	addiu	$29,$29,-8
	sw	$31,4($29)
	xor	$2,$2,$2
while:
	lw	$8,0($4)
	bltz	$8,endwhile
	addiu	$2,$2,1
	addiu	$4,$4,4
	j while
endwhile:
# epilogue
	lw	$31,4($29)
	addiu	$29,$29,8
	jr	$31

sumtab:
# prologue
	addiu	$29,$29,-12
	sw	$31,8($29)
	xor	$2,$2,$2
# boucle
while1:
	lw	$8,0($4)
	bltz	$8,endwhile1
	addu	$2,$2,$8
	addiu	$4,$4,4
	j 	while1
endwhile1:
	# epilogue
	lw	$31,8($29)
	addiu	$29,$29,12
	jr	$31


